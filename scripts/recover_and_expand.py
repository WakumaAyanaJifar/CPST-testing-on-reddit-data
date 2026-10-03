#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPST-FoMO: recovery and author expansion (v6.1)
===============================================

Run this against the database produced by fomo_collector_v6.py. It does four
things, in order, and each one is resumable:

  1. DIAGNOSE     report the current state of the corpus
  2. RECOVER      rebuild the author_hash -> username map by bulk-refetching the
                  items already collected (~100 items per API request)
  3. RECLASSIFY   fix the cohort definitions that v6.0 got wrong:
                      seed        = author of an item with a keyword hit
                      participant = commented in a seed thread, no keyword hit
                      control     = matched sample drawn from participants
  4. EXPAND       stage C, the author-window collection that never ran

Why the cohorts changed
    v6.0 marked every commenter in a seed thread as a seed author, which
    inflated the seed cohort and left no clean comparison group. Participants
    are in-window and community-matched by construction, which makes them a
    better control pool than anything .new() can produce for a historical window.

The name map
    Stage C needs real usernames. This script writes them to a separate file,
    encrypted if `cryptography` is installed. That file is identifiable data:
    name it in your ethics protocol and delete it when collection ends. The
    corpus itself never holds a username.

Usage
    python recover_and_expand.py --diagnose
    python recover_and_expand.py --recover
    python recover_and_expand.py --reclassify
    python recover_and_expand.py --expand --cohort seed
    python recover_and_expand.py --expand --cohort control
    python recover_and_expand.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import praw
import prawcore

try:
    from cryptography.fernet import Fernet
    HAVE_CRYPTO = True
except ImportError:
    HAVE_CRYPTO = False


# ---------------------------------------------------------------- config

# Data folder: set CPST_BASE_DIR to override the default location.
BASE_DIR = os.environ.get(
    "CPST_BASE_DIR",
    os.path.join(os.path.expanduser("~"), "Documents", "CPST_FOMO_v6"))
DB_PATH = os.path.join(BASE_DIR, "cpst_fomo.db")
SALT_PATH = os.path.join(BASE_DIR, "author_salt.key")
NAMEMAP_PATH = os.path.join(BASE_DIR, "author_namemap.enc")
NAMEKEY_PATH = os.path.join(BASE_DIR, "author_namemap.key")
LOG_PATH = os.path.join(BASE_DIR, "expansion.log")

WINDOW_START = "2023-01-01"
WINDOW_END = "2025-12-31"
AUTHOR_HISTORY_LIMIT = 1000
MIN_RECORDS_PER_AUTHOR = 30
CONTROL_RATIO = 1.0
INFO_BATCH = 100          # reddit.info() ceiling

USER_AGENT = "CPST-FoMO-Research/6.1 (USTB; academic, non-commercial)"

WINDOW_START_TS = datetime.fromisoformat(WINDOW_START).replace(
    tzinfo=timezone.utc).timestamp()
WINDOW_END_TS = datetime.fromisoformat(WINDOW_END).replace(
    tzinfo=timezone.utc).timestamp()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)])


# ---------------------------------------------------------------- pseudonymiser

class Pseudonymiser:
    def __init__(self, salt_path: str = SALT_PATH):
        if not os.path.exists(salt_path):
            sys.exit(f"Salt file missing: {salt_path}\n"
                     "Without the original salt the hashes cannot be reproduced.")
        with open(salt_path, "rb") as f:
            self.salt = f.read()

    def hash(self, username: Optional[str]) -> Optional[str]:
        if not username or username in ("[deleted]", "None", "AutoModerator"):
            return None
        return hashlib.blake2b(username.lower().encode("utf-8"),
                               salt=self.salt[:16], digest_size=16).hexdigest()


# ---------------------------------------------------------------- name map

class NameMap:
    """hash -> username. Encrypted at rest when `cryptography` is available."""

    def __init__(self):
        self.data: Dict[str, str] = {}
        self._fernet = None
        if HAVE_CRYPTO:
            if os.path.exists(NAMEKEY_PATH):
                with open(NAMEKEY_PATH, "rb") as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                with open(NAMEKEY_PATH, "wb") as f:
                    f.write(key)
                try:
                    os.chmod(NAMEKEY_PATH, 0o600)
                except OSError:
                    pass
            self._fernet = Fernet(key)
        else:
            logging.warning(
                "`cryptography` not installed, the name map will be stored as "
                "plain JSON. Install it with: python -m pip install cryptography")
        self.load()

    def load(self):
        if not os.path.exists(NAMEMAP_PATH):
            return
        with open(NAMEMAP_PATH, "rb") as f:
            blob = f.read()
        if self._fernet:
            try:
                blob = self._fernet.decrypt(blob)
            except Exception:
                logging.error("Could not decrypt the name map. Wrong key file?")
                return
        self.data = json.loads(blob.decode("utf-8"))
        logging.info("name map loaded: %d entries", len(self.data))

    def save(self):
        blob = json.dumps(self.data).encode("utf-8")
        if self._fernet:
            blob = self._fernet.encrypt(blob)
        with open(NAMEMAP_PATH, "wb") as f:
            f.write(blob)
        try:
            os.chmod(NAMEMAP_PATH, 0o600)
        except OSError:
            pass
        logging.info("name map saved: %d entries", len(self.data))


# ---------------------------------------------------------------- reddit

def get_reddit() -> praw.Reddit:
    for var in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"):
        if var not in os.environ:
            sys.exit(f"Set {var} in the environment before running.")
    r = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=USER_AGENT,
        ratelimit_seconds=600,
        check_for_async=False,
    )
    r.read_only = True
    return r


def batched(seq: List, n: int) -> Iterable[List]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ---------------------------------------------------------------- 1. diagnose

def diagnose(conn: sqlite3.Connection):
    q = lambda s, *a: conn.execute(s, a).fetchone()[0]

    print("\n" + "=" * 70)
    print("CORPUS STATE")
    print("=" * 70)
    n_posts = q("SELECT COUNT(*) FROM items WHERE kind='post'")
    n_comments = q("SELECT COUNT(*) FROM items WHERE kind='comment'")
    print(f"items                  {q('SELECT COUNT(*) FROM items'):>10,}")
    print(f"  posts                {n_posts:>10,}")
    print(f"  comments             {n_comments:>10,}")
    print(f"threads                {q('SELECT COUNT(*) FROM threads'):>10,}")
    print(f"keyword hits           {q('SELECT COUNT(*) FROM keyword_hits'):>10,}")
    print(f"authors (all)          {q('SELECT COUNT(*) FROM authors'):>10,}")

    print("\nby stage")
    for stage, n in conn.execute(
            "SELECT stage, COUNT(*) FROM items GROUP BY stage ORDER BY 2 DESC"):
        print(f"  {stage:<22}{n:>10,}")

    print("\nby cohort")
    for cohort, n in conn.execute(
            "SELECT COALESCE(cohort,'(none)'), COUNT(*) FROM authors GROUP BY 1"):
        print(f"  {cohort:<22}{n:>10,}")

    print("\nauthor volume distribution")
    for label, lo in (("1 record", 1), (">=5", 5), (">=10", 10),
                      (">=30", 30), (">=50", 50), (">=100", 100)):
        n = q("SELECT COUNT(*) FROM authors WHERE n_items >= ?", lo)
        print(f"  {label:<22}{n:>10,}")

    n_done = q("SELECT COUNT(*) FROM authors WHERE expanded_at IS NOT NULL")
    n_todo = q("SELECT COUNT(*) FROM authors WHERE expanded_at IS NULL")
    print("\nexpansion status")
    print(f"  expanded             {n_done:>10,}")
    print(f"  pending              {n_todo:>10,}")

    print("\nby community")
    for sub, n in conn.execute(
            "SELECT subreddit, COUNT(*) FROM items GROUP BY 1 ORDER BY 2 DESC"):
        print(f"  {sub:<22}{n:>10,}")

    print("\ntemporal coverage")
    row = conn.execute("SELECT MIN(created_utc), MAX(created_utc) FROM items").fetchone()
    if row[0]:
        print(f"  {datetime.fromtimestamp(row[0], timezone.utc):%Y-%m-%d}"
              f"  to  {datetime.fromtimestamp(row[1], timezone.utc):%Y-%m-%d}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------- 2. recover

def recover_names(conn: sqlite3.Connection, reddit, pseudo: Pseudonymiser,
                  namemap: NameMap):
    """Rebuild hash -> username by bulk-refetching collected items."""
    cur = conn.execute(
        "SELECT id, kind FROM items WHERE author_hash IS NOT NULL")
    rows = cur.fetchall()
    fullnames = [f"{'t3_' if k == 'post' else 't1_'}{i}" for i, k in rows]

    known = set(namemap.data)
    logging.info("recovering names from %d items (%d hashes already known)",
                 len(fullnames), len(known))

    found = 0
    for bi, batch in enumerate(batched(fullnames, INFO_BATCH), 1):
        try:
            for obj in reddit.info(fullnames=batch):
                author = getattr(obj, "author", None)
                if not author:
                    continue
                name = str(author)
                h = pseudo.hash(name)
                if h and h not in namemap.data:
                    namemap.data[h] = name
                    found += 1
        except prawcore.exceptions.PrawcoreException as e:
            logging.warning("info() batch %d failed: %s", bi, e)
            continue

        if bi % 25 == 0:
            namemap.save()
            logging.info("  batch %d/%d, %d names recovered",
                         bi, (len(fullnames) + INFO_BATCH - 1) // INFO_BATCH, found)

    namemap.save()

    total_authors = conn.execute(
        "SELECT COUNT(*) FROM authors").fetchone()[0]
    covered = sum(1 for (h,) in conn.execute("SELECT author_hash FROM authors")
                  if h in namemap.data)
    logging.info("recovery complete: %d names, covering %d/%d authors (%.1f%%)",
                 len(namemap.data), covered, total_authors,
                 100 * covered / max(total_authors, 1))
    logging.info("uncovered authors are deleted, suspended or shadowbanned accounts")


# ---------------------------------------------------------------- 3. reclassify

def reclassify(conn: sqlite3.Connection):
    """Fix the cohort definitions. v6.0 called every thread commenter a seed."""
    logging.info("reclassifying cohorts")

    conn.execute("""
        UPDATE authors SET cohort = 'participant'
        WHERE author_hash NOT IN (
            SELECT DISTINCT i.author_hash
            FROM items i JOIN keyword_hits k ON k.item_id = i.id
            WHERE i.author_hash IS NOT NULL)
    """)
    conn.execute("""
        UPDATE authors SET cohort = 'seed'
        WHERE author_hash IN (
            SELECT DISTINCT i.author_hash
            FROM items i JOIN keyword_hits k ON k.item_id = i.id
            WHERE i.author_hash IS NOT NULL)
    """)
    conn.commit()

    n_seed = conn.execute("SELECT COUNT(*) FROM authors WHERE cohort='seed'").fetchone()[0]
    n_part = conn.execute("SELECT COUNT(*) FROM authors WHERE cohort='participant'").fetchone()[0]
    logging.info("  seed: %d, participant: %d", n_seed, n_part)

    # --- matched control selection ------------------------------------
    # Match on block and on a volume bin. Participants are in-window and
    # community-matched already, so matching on volume removes the remaining
    # obvious confound (prolific users differ from occasional ones).
    logging.info("selecting matched controls")

    def bin_of(n: int) -> int:
        for i, edge in enumerate((1, 2, 5, 10, 25, 50, 100)):
            if n <= edge:
                return i
        return 7

    seed_strata = defaultdict(int)
    for block, n in conn.execute(
            "SELECT seed_block, n_items FROM authors WHERE cohort='seed'"):
        seed_strata[(block, bin_of(n or 0))] += 1

    pool = defaultdict(list)
    for h, block, n in conn.execute(
            "SELECT author_hash, seed_block, n_items FROM authors WHERE cohort='participant'"):
        pool[(block, bin_of(n or 0))].append(h)

    chosen: List[str] = []
    shortfall = 0
    for stratum, need in seed_strata.items():
        need = int(need * CONTROL_RATIO)
        available = pool.get(stratum, [])
        take = available[:need]
        chosen.extend(take)
        if len(take) < need:
            shortfall += need - len(take)

    conn.executemany("UPDATE authors SET cohort='control' WHERE author_hash=?",
                     [(h,) for h in chosen])
    conn.commit()

    logging.info("  controls selected: %d (shortfall %d strata slots)",
                 len(chosen), shortfall)
    if shortfall:
        logging.warning("  shortfall means some seed strata have no participant "
                        "match; report this in the methods rather than "
                        "silently accepting unbalanced groups")


# ---------------------------------------------------------------- 4. expand

def record_post(conn, p, stage: str, pseudo: Pseudonymiser, block_of: Dict[str, str]):
    sub = p.subreddit.display_name
    raw = p.selftext or ""
    conn.execute("INSERT OR IGNORE INTO items VALUES "
                 "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        p.id, "post", sub, block_of.get(sub, "other"),
        pseudo.hash(str(p.author) if p.author else None),
        float(p.created_utc), p.title, raw, None, len(raw),
        p.score, p.upvote_ratio, p.num_comments,
        None, f"t3_{p.id}", 0, None,
        p.permalink, int(bool(p.is_self)), getattr(p, "domain", None),
        int(bool(p.over_18)), float(p.edited) if p.edited else None,
        stage, time.time()))


def record_comment(conn, c, stage: str, pseudo: Pseudonymiser, block_of: Dict[str, str]):
    sub = c.subreddit.display_name
    raw = c.body or ""
    conn.execute("INSERT OR IGNORE INTO items VALUES "
                 "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        c.id, "comment", sub, block_of.get(sub, "other"),
        pseudo.hash(str(c.author) if c.author else None),
        float(c.created_utc), None, raw, None, len(raw),
        c.score, None, None,
        c.parent_id, c.link_id, getattr(c, "depth", None), None,
        c.permalink, None, None, None,
        float(c.edited) if c.edited else None,
        stage, time.time()))


def expand(conn: sqlite3.Connection, reddit, pseudo: Pseudonymiser,
           namemap: NameMap, cohort: str):
    """Stage C. This is the step that creates the author-window corpus."""
    block_of = dict(conn.execute(
        "SELECT DISTINCT subreddit, block FROM items WHERE block IS NOT NULL"))

    todo = [h for (h,) in conn.execute(
        "SELECT author_hash FROM authors WHERE cohort=? AND expanded_at IS NULL",
        (cohort,)) if h in namemap.data]

    logging.info("stage C (%s): %d authors to expand", cohort, len(todo))
    if not todo:
        logging.warning("nothing to expand. Did you run --recover and --reclassify?")
        return

    t0 = time.time()
    for i, h in enumerate(todo, 1):
        name = namemap.data[h]
        try:
            r = reddit.redditor(name)
            for listing, is_post in ((r.submissions.new(limit=AUTHOR_HISTORY_LIMIT), True),
                                     (r.comments.new(limit=AUTHOR_HISTORY_LIMIT), False)):
                for item in listing:
                    ts = float(item.created_utc)
                    if ts < WINDOW_START_TS:
                        break            # listings are reverse-chronological
                    if ts > WINDOW_END_TS:
                        continue
                    if is_post:
                        record_post(conn, item, f"C_author_{cohort}", pseudo, block_of)
                    else:
                        record_comment(conn, item, f"C_author_{cohort}", pseudo, block_of)
            conn.execute("UPDATE authors SET expanded_at=? WHERE author_hash=?",
                         (time.time(), h))
        except prawcore.exceptions.NotFound:
            conn.execute("UPDATE authors SET expanded_at=? WHERE author_hash=?",
                         (time.time(), h))          # suspended or deleted
        except prawcore.exceptions.PrawcoreException as e:
            logging.warning("expansion failed for one author: %s", e)
            continue

        if i % 25 == 0:
            conn.commit()
            rate = i / (time.time() - t0)
            eta = (len(todo) - i) / rate / 60
            n = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            logging.info("  %d/%d authors | %d items | %.1f/min | ETA %.0f min",
                         i, len(todo), n, rate * 60, eta)

    conn.commit()
    logging.info("stage C (%s) complete", cohort)


def refresh_author_stats(conn: sqlite3.Connection):
    conn.executescript("""
    UPDATE authors SET
        n_items      = (SELECT COUNT(*)                  FROM items i WHERE i.author_hash = authors.author_hash),
        first_seen   = (SELECT MIN(created_utc)          FROM items i WHERE i.author_hash = authors.author_hash),
        last_seen    = (SELECT MAX(created_utc)          FROM items i WHERE i.author_hash = authors.author_hash),
        n_subreddits = (SELECT COUNT(DISTINCT subreddit) FROM items i WHERE i.author_hash = authors.author_hash);
    """)
    conn.commit()


# ---------------------------------------------------------------- main

def _args(ap):
    """Parsed arguments, safe under Jupyter.

    In a notebook sys.argv carries the kernel's own flags, and argparse's
    prefix matching binds --f=<kernel>.json to --features without complaint,
    so the script runs against the kernel connection file. Under ipykernel we
    therefore ignore sys.argv entirely and use the defaults; pass options from
    a terminal instead, or edit the constants at the top of the file.
    """
    in_notebook = ("ipykernel" in sys.modules) or any(
        ("ipykernel_launcher" in a) or ("kernel-" in a and a.endswith(".json"))
        for a in sys.argv)
    return ap.parse_args([] if in_notebook else sys.argv[1:])


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--recover", action="store_true")
    ap.add_argument("--reclassify", action="store_true")
    ap.add_argument("--expand", action="store_true")
    ap.add_argument("--cohort", default="seed", choices=["seed", "control"])
    ap.add_argument("--all", action="store_true")
    args = _args(ap)

    if not os.path.exists(args.db):
        sys.exit(f"Database not found: {args.db}")

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL")

    if args.diagnose and not args.all:
        refresh_author_stats(conn)
        diagnose(conn)
        conn.close()
        return

    pseudo = Pseudonymiser()
    namemap = NameMap()
    reddit = get_reddit()

    if args.all or args.recover:
        recover_names(conn, reddit, pseudo, namemap)
    if args.all or args.reclassify:
        refresh_author_stats(conn)
        reclassify(conn)
    if args.all or args.expand:
        for cohort in (["seed", "control"] if args.all else [args.cohort]):
            expand(conn, reddit, pseudo, namemap, cohort)

    refresh_author_stats(conn)
    diagnose(conn)
    conn.close()


if __name__ == "__main__":
    main()
