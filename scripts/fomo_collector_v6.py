#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPST-FoMO Corpus Collector v6.0
===============================

Four-stage collection designed so that all four CPST spaces are recoverable.

    Stage A  seed discovery     keyword search, used ONLY to identify candidate authors
    Stage B  thread completion  full comment tree of every seed submission (parent_id, depth)
    Stage C  author expansion   complete public history of seed authors in the window
    Stage D  control cohort     matched authors with no seed hit, same communities

Design rules, each of which answers a reviewer objection:

  1. No quality filters at collection. Nothing is discarded for score or length.
     Filtering happens at analysis where it is visible, documented and reversible.
     (v5 dropped every downvoted record, biasing against badly-received disclosure.)
  2. Raw text is preserved alongside cleaned text. URLs are kept, since which
     platform a user links to is Cyber-space evidence.
  3. parent_id, link_id and depth are stored. Without them there is no Social space.
  4. Author names are salted-hashed at write time. The salt lives outside the DB.
  5. Keyword provenance goes in a separate table, never in the item row, so it
     cannot leak into a feature matrix. theoretical_domain is a deterministic
     function of the keyword and is therefore NOT a label.
  6. Every record carries its collection stage, so selection bias can be modelled
     rather than denied.

COMPLIANCE NOTE
    Reddit's documentation states that the Reddit for Researchers (RFR) programme
    is the authorised route for academic research. Use this collector for pilot and
    method-development work, and apply to RFR for the confirmatory corpus. The
    schema below is deliberately compatible with an RFR bulk load: RFR supplies
    hashed user ids, which map onto author_hash here.

CREDENTIALS
    Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET as environment variables.
    Never hard-code them. If they have ever appeared in a file, a notebook or a
    chat window, rotate them at reddit.com/prefs/apps before running this.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterable, List, Optional, Set

import praw
import prawcore


# ==========================================================================
# CONFIGURATION
# ==========================================================================

@dataclass
class Config:
    # --- credentials -----------------------------------------------------
    CLIENT_ID: str = field(default_factory=lambda: os.environ["REDDIT_CLIENT_ID"])
    CLIENT_SECRET: str = field(default_factory=lambda: os.environ["REDDIT_CLIENT_SECRET"])
    USER_AGENT: str = "CPST-FoMO-Research/6.0 (USTB; academic, non-commercial)"

    # --- sampling frame --------------------------------------------------
    # Two blocks, kept explicitly separate. The contrast between them is the
    # comparative design: FoMO expressed inside an active disengagement attempt
    # versus inside a general anxiety context.
    ANXIETY_BLOCK: List[str] = field(default_factory=lambda: [
        "Anxiety", "socialanxiety", "healthanxiety", "panicdisorder",
    ])
    DISENGAGEMENT_BLOCK: List[str] = field(default_factory=lambda: [
        "nosurf", "StopGaming", "digitalminimalism", "DigitalDetox",
    ])

    # --- observation window ----------------------------------------------
    # Fixed window. Do not extend it mid-study; a moving window makes the
    # temporal confound unanalysable.
    WINDOW_START: str = "2023-01-01"
    WINDOW_END: str = "2025-12-31"

    # --- stage parameters ------------------------------------------------
    SEED_POSTS_PER_KEYWORD: int = 100
    MAX_MORE_EXPANSIONS: int = 32       # replace_more cap; 0 in v5 destroyed the tree
    AUTHOR_HISTORY_LIMIT: int = 1000    # Reddit's own listing ceiling
    MIN_RECORDS_PER_AUTHOR: int = 30    # threshold for per-author circadian estimation
    TARGET_SEED_AUTHORS: int = 6000
    CONTROL_RATIO: float = 1.0          # one matched control per seed author

    # --- storage ---------------------------------------------------------
    BASE_DIR: str = field(default_factory=lambda: os.environ.get(
        "CPST_BASE_DIR",
        os.path.join(os.path.expanduser("~"), "Documents", "CPST_FOMO_v6")))

    # --- keyword taxonomy (discovery only, never a label) ----------------
    KEYWORDS: Dict[str, List[str]] = field(default_factory=lambda: {
        "Core": [
            "fear of missing out", "fomo", "missing out on",
            "afraid of missing", "don't want to miss",
        ],
        "Comparison": [
            "everyone else is", "comparing myself", "falling behind",
            "not keeping up", "everyone has their life together",
            "I'm the only one", "wasn't invited", "left out",
            "wish I was there", "could have been",
        ],
        "Compulsive": [
            "can't stop checking", "compulsively checking", "every few minutes",
            "keep opening", "without thinking", "lost track of time",
            "hours disappeared", "spent all day scrolling",
            "bedtime procrastination", "wake up and check",
        ],
        "Platform": [
            "left on seen", "story views", "instagram vs reality",
            "just one more video", "algorithm got me", "streak anxiety",
            "doomscrolling", "not in the loop",
        ],
        "Somatic": [
            "chest tight", "heart racing", "can't breathe",
            "sick to my stomach", "couldn't sleep", "up until 3am",
        ],
        "Recovery": [
            "digital detox", "deleted the app", "quit instagram",
            "taking a break from", "logging off", "relapsed",
        ],
    })

    def __post_init__(self):
        os.makedirs(self.BASE_DIR, exist_ok=True)
        self.DB_PATH = os.path.join(self.BASE_DIR, "cpst_fomo.db")
        self.LOG_PATH = os.path.join(self.BASE_DIR, "collection.log")
        self.SALT_PATH = os.path.join(self.BASE_DIR, "author_salt.key")
        self.METADATA_PATH = os.path.join(self.BASE_DIR, "metadata.json")

        self.SUBREDDITS = self.ANXIETY_BLOCK + self.DISENGAGEMENT_BLOCK
        self.BLOCK_OF = {s: "anxiety" for s in self.ANXIETY_BLOCK}
        self.BLOCK_OF.update({s: "disengagement" for s in self.DISENGAGEMENT_BLOCK})

        self.FLAT_KEYWORDS = [(kw, dom)
                              for dom, kws in self.KEYWORDS.items()
                              for kw in kws]

        self.WINDOW_START_TS = datetime.fromisoformat(
            self.WINDOW_START).replace(tzinfo=timezone.utc).timestamp()
        self.WINDOW_END_TS = datetime.fromisoformat(
            self.WINDOW_END).replace(tzinfo=timezone.utc).timestamp()


# ==========================================================================
# PSEUDONYMISATION
# ==========================================================================

class Pseudonymiser:
    """Salted hash of usernames. The salt is generated once and kept outside
    the database. Losing the salt makes re-identification impossible, which is
    the point: hold it only as long as the ethics protocol requires."""

    def __init__(self, salt_path: str):
        if os.path.exists(salt_path):
            with open(salt_path, "rb") as f:
                self.salt = f.read()
        else:
            self.salt = secrets.token_bytes(32)
            with open(salt_path, "wb") as f:
                f.write(self.salt)
            os.chmod(salt_path, 0o600)
            logging.warning("New author salt written to %s. Back it up separately "
                            "from the database, and delete it at project end.", salt_path)

    def hash(self, username: Optional[str]) -> Optional[str]:
        if not username or username in ("[deleted]", "None", "AutoModerator"):
            return None
        return hashlib.blake2b(username.lower().encode("utf-8"),
                               salt=self.salt[:16], digest_size=16).hexdigest()


# ==========================================================================
# DATABASE
# ==========================================================================

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

-- Every post and comment. No filtering applied here.
CREATE TABLE IF NOT EXISTS items (
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL CHECK(kind IN ('post','comment')),
    subreddit       TEXT NOT NULL,          -- always display_name
    block           TEXT,                   -- anxiety | disengagement | other
    author_hash     TEXT,                   -- NULL for deleted / AutoModerator
    created_utc     REAL NOT NULL,          -- epoch seconds, not a string
    title           TEXT,                   -- posts only
    body_raw        TEXT,                   -- verbatim, URLs intact
    body_clean      TEXT,                   -- derived, for modelling
    n_chars         INTEGER,
    score           INTEGER,
    upvote_ratio    REAL,                   -- posts only, NULL for comments
    num_comments    INTEGER,                -- posts only, NULL for comments
    -- Social space
    parent_id       TEXT,                   -- t1_/t3_ fullname, comments only
    link_id         TEXT,                   -- submission fullname
    depth           INTEGER,                -- reply depth in tree
    is_op_reply     INTEGER,                -- comment author == submission author
    -- Cyber space
    permalink       TEXT,
    is_self         INTEGER,
    domain          TEXT,                   -- link target for link posts
    over_18         INTEGER,
    edited          REAL,
    -- provenance
    stage           TEXT NOT NULL,          -- A_seed | B_thread | C_author | D_control
    collected_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_author ON items(author_hash);
CREATE INDEX IF NOT EXISTS idx_items_link   ON items(link_id);
CREATE INDEX IF NOT EXISTS idx_items_sub    ON items(subreddit);
CREATE INDEX IF NOT EXISTS idx_items_time   ON items(created_utc);
CREATE INDEX IF NOT EXISTS idx_items_stage  ON items(stage);

-- Keyword provenance lives HERE, never on the item row, so it cannot be
-- joined into a feature matrix by accident.
CREATE TABLE IF NOT EXISTS keyword_hits (
    item_id     TEXT NOT NULL,
    keyword     TEXT NOT NULL,
    domain      TEXT NOT NULL,
    PRIMARY KEY (item_id, keyword)
);

-- Author-window is the CPST unit of analysis.
CREATE TABLE IF NOT EXISTS authors (
    author_hash   TEXT PRIMARY KEY,
    cohort        TEXT,                     -- seed | control
    seed_block    TEXT,                     -- block the author was seeded from
    n_items       INTEGER DEFAULT 0,
    first_seen    REAL,
    last_seen     REAL,
    n_subreddits  INTEGER,
    expanded_at   REAL                      -- NULL until stage C has run
);

CREATE TABLE IF NOT EXISTS threads (
    link_id        TEXT PRIMARY KEY,
    subreddit      TEXT,
    n_collected    INTEGER,
    more_remaining INTEGER,                 -- unexpanded branches, for honesty
    completed_at   REAL
);

CREATE TABLE IF NOT EXISTS run_log (
    ts      REAL,
    stage   TEXT,
    event   TEXT,
    detail  TEXT
);
"""


class Store:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._pending = 0

    def upsert_item(self, rec: Dict) -> bool:
        cols = ", ".join(rec)
        ph = ", ".join("?" * len(rec))
        try:
            self.conn.execute(
                f"INSERT OR IGNORE INTO items ({cols}) VALUES ({ph})",
                list(rec.values()))
            self._pending += 1
            if self._pending >= 500:
                self.flush()
            return True
        except sqlite3.Error as e:
            logging.error("insert failed for %s: %s", rec.get("id"), e)
            return False

    def add_keyword_hit(self, item_id: str, keyword: str, domain: str):
        self.conn.execute(
            "INSERT OR IGNORE INTO keyword_hits (item_id, keyword, domain) VALUES (?,?,?)",
            (item_id, keyword, domain))
        self._pending += 1

    def note_author(self, author_hash: str, cohort: str, block: Optional[str]):
        self.conn.execute(
            "INSERT OR IGNORE INTO authors (author_hash, cohort, seed_block) VALUES (?,?,?)",
            (author_hash, cohort, block))
        self._pending += 1

    def mark_expanded(self, author_hash: str):
        self.conn.execute("UPDATE authors SET expanded_at=? WHERE author_hash=?",
                          (time.time(), author_hash))
        self._pending += 1

    def note_thread(self, link_id, subreddit, n, more):
        self.conn.execute(
            "INSERT OR REPLACE INTO threads VALUES (?,?,?,?,?)",
            (link_id, subreddit, n, more, time.time()))
        self._pending += 1

    def log(self, stage: str, event: str, detail: str = ""):
        self.conn.execute("INSERT INTO run_log VALUES (?,?,?,?)",
                          (time.time(), stage, event, detail))
        self._pending += 1

    def pending_authors(self, cohort: str, limit: int) -> List[str]:
        cur = self.conn.execute(
            "SELECT author_hash FROM authors WHERE cohort=? AND expanded_at IS NULL LIMIT ?",
            (cohort, limit))
        return [r[0] for r in cur.fetchall()]

    def author_count(self, cohort: str) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM authors WHERE cohort=?", (cohort,))
        return cur.fetchone()[0]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]

    def refresh_author_stats(self):
        """Recompute per-author aggregates. Run once after collection."""
        self.conn.executescript("""
        UPDATE authors SET
            n_items      = (SELECT COUNT(*)            FROM items i WHERE i.author_hash = authors.author_hash),
            first_seen   = (SELECT MIN(created_utc)    FROM items i WHERE i.author_hash = authors.author_hash),
            last_seen    = (SELECT MAX(created_utc)    FROM items i WHERE i.author_hash = authors.author_hash),
            n_subreddits = (SELECT COUNT(DISTINCT subreddit) FROM items i WHERE i.author_hash = authors.author_hash);
        """)
        self.conn.commit()

    def flush(self):
        self.conn.commit()
        self._pending = 0

    def close(self):
        self.flush()
        self.conn.close()


# ==========================================================================
# COLLECTOR
# ==========================================================================

URL_RE = re.compile(r"https?://\S+")
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
WS_RE = re.compile(r"\s+")


class Collector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.BASE_DIR, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(cfg.LOG_PATH, encoding="utf-8"),
                      logging.StreamHandler(sys.stdout)])

        self.store = Store(cfg.DB_PATH)
        self.pseudo = Pseudonymiser(cfg.SALT_PATH)
        self.stats = defaultdict(int)

        self.reddit = praw.Reddit(
            client_id=cfg.CLIENT_ID,
            client_secret=cfg.CLIENT_SECRET,
            user_agent=cfg.USER_AGENT,
            ratelimit_seconds=600,      # let PRAW handle backoff; no manual sleeps
            check_for_async=False,
        )
        self.reddit.read_only = True

    # ---------------- helpers ----------------

    @staticmethod
    def _clean(text: str) -> str:
        """Cleaned variant for modelling. The raw text is stored separately;
        never let this be the only copy."""
        if not text:
            return ""
        text = MD_LINK_RE.sub(r"\1", text)
        text = URL_RE.sub(" <URL> ", text)
        text = re.sub(r"[#*_~`>|^]", "", text)
        return WS_RE.sub(" ", text).strip()

    def _in_window(self, ts: float) -> bool:
        return self.cfg.WINDOW_START_TS <= ts <= self.cfg.WINDOW_END_TS

    def _record_post(self, p, stage: str) -> Optional[Dict]:
        try:
            ts = float(p.created_utc)
            sub = p.subreddit.display_name          # always true casing
            raw = (p.selftext or "")
            rec = {
                "id": p.id,
                "kind": "post",
                "subreddit": sub,
                "block": self.cfg.BLOCK_OF.get(sub, "other"),
                "author_hash": self.pseudo.hash(str(p.author) if p.author else None),
                "created_utc": ts,
                "title": p.title,
                "body_raw": raw,
                "body_clean": self._clean(f"{p.title} {raw}"),
                "n_chars": len(raw),
                "score": p.score,
                "upvote_ratio": p.upvote_ratio,
                "num_comments": p.num_comments,
                "parent_id": None,
                "link_id": f"t3_{p.id}",
                "depth": 0,
                "is_op_reply": None,
                "permalink": p.permalink,
                "is_self": int(bool(p.is_self)),
                "domain": getattr(p, "domain", None),
                "over_18": int(bool(p.over_18)),
                "edited": float(p.edited) if p.edited else None,
                "stage": stage,
                "collected_at": time.time(),
            }
            self.store.upsert_item(rec)
            self.stats[f"post_{stage}"] += 1
            return rec
        except Exception as e:
            logging.warning("post record failed: %s", e)
            return None

    def _record_comment(self, c, stage: str, op_author: Optional[str] = None) -> Optional[Dict]:
        try:
            ts = float(c.created_utc)
            sub = c.subreddit.display_name
            raw = c.body or ""
            author = str(c.author) if c.author else None
            rec = {
                "id": c.id,
                "kind": "comment",
                "subreddit": sub,
                "block": self.cfg.BLOCK_OF.get(sub, "other"),
                "author_hash": self.pseudo.hash(author),
                "created_utc": ts,
                "title": None,
                "body_raw": raw,
                "body_clean": self._clean(raw),
                "n_chars": len(raw),
                "score": c.score,
                "upvote_ratio": None,          # comments have none; do NOT write 0.0
                "num_comments": None,
                "parent_id": c.parent_id,
                "link_id": c.link_id,
                "depth": getattr(c, "depth", None),
                "is_op_reply": int(author == op_author) if (author and op_author) else None,
                "permalink": c.permalink,
                "is_self": None,
                "domain": None,
                "over_18": None,
                "edited": float(c.edited) if c.edited else None,
                "stage": stage,
                "collected_at": time.time(),
            }
            self.store.upsert_item(rec)
            self.stats[f"comment_{stage}"] += 1
            return rec
        except Exception as e:
            logging.warning("comment record failed: %s", e)
            return None

    # ---------------- Stage A: seed discovery ----------------

    def stage_a_seed(self):
        """Keyword search. Its ONLY purpose is to identify candidate authors.
        The records it returns are kept, but they are not the corpus."""
        logging.info("STAGE A: seed discovery")
        self.store.log("A", "start")

        for sub_name in self.cfg.SUBREDDITS:
            sub = self.reddit.subreddit(sub_name)
            block = self.cfg.BLOCK_OF[sub_name]

            for keyword, domain in self.cfg.FLAT_KEYWORDS:
                query = f'"{keyword}"' if " " in keyword else keyword
                try:
                    results = sub.search(query, sort="new", time_filter="all",
                                         limit=self.cfg.SEED_POSTS_PER_KEYWORD)
                    for p in results:
                        if not self._in_window(float(p.created_utc)):
                            continue
                        rec = self._record_post(p, "A_seed")
                        if rec:
                            self.store.add_keyword_hit(p.id, keyword, domain)
                            if rec["author_hash"]:
                                self.store.note_author(rec["author_hash"], "seed", block)
                except prawcore.exceptions.PrawcoreException as e:
                    logging.warning("search failed r/%s '%s': %s", sub_name, keyword, e)
                    continue

            self.store.flush()
            logging.info("  r/%-18s seeds so far: %d authors, %d items",
                         sub_name, self.store.author_count("seed"), self.store.count())

        self.store.log("A", "done", f"{self.store.author_count('seed')} seed authors")

    # ---------------- Stage B: thread completion ----------------

    def stage_b_threads(self):
        """Full comment tree of every seed submission. This is the stage v5
        omitted, and the reason the Social space was empty."""
        logging.info("STAGE B: thread completion")
        self.store.log("B", "start")

        cur = self.store.conn.execute(
            "SELECT id FROM items WHERE kind='post' AND stage='A_seed' "
            "AND link_id NOT IN (SELECT link_id FROM threads)")
        post_ids = [r[0] for r in cur.fetchall()]
        logging.info("  %d threads to expand", len(post_ids))

        for i, pid in enumerate(post_ids, 1):
            try:
                sub = self.reddit.submission(id=pid)
                op = str(sub.author) if sub.author else None
                more = sub.comments.replace_more(limit=self.cfg.MAX_MORE_EXPANSIONS)
                n = 0
                for c in sub.comments.list():
                    if self._record_comment(c, "B_thread", op_author=op):
                        n += 1
                        if c.author:
                            h = self.pseudo.hash(str(c.author))
                            if h:
                                self.store.note_author(
                                    h, "seed", self.cfg.BLOCK_OF.get(
                                        sub.subreddit.display_name, "other"))
                self.store.note_thread(f"t3_{pid}", sub.subreddit.display_name,
                                       n, len(more) if more else 0)
            except prawcore.exceptions.PrawcoreException as e:
                logging.warning("thread %s failed: %s", pid, e)
                continue

            if i % 50 == 0:
                self.store.flush()
                logging.info("  %d/%d threads, %d items total", i, len(post_ids),
                             self.store.count())

        self.store.log("B", "done")

    # ---------------- Stage C: author expansion ----------------

    def stage_c_authors(self, cohort: str = "seed"):
        """Complete public history of each author inside the window, across ALL
        communities. This is where Cyber, Physical and Social actually come from."""
        logging.info("STAGE C: author expansion (%s cohort)", cohort)
        self.store.log("C", "start", cohort)

        todo = self.store.pending_authors(cohort, self.cfg.TARGET_SEED_AUTHORS)
        logging.info("  %d authors to expand", len(todo))

        # Reverse lookup is impossible from the hash, so stage C must run in the
        # same process as the stage that discovered the author, or the mapping
        # must be cached. We cache it on the instance.
        for i, h in enumerate(todo, 1):
            name = self._name_cache.get(h)
            if not name:
                continue
            try:
                r = self.reddit.redditor(name)
                for listing in (r.submissions.new(limit=self.cfg.AUTHOR_HISTORY_LIMIT),
                                r.comments.new(limit=self.cfg.AUTHOR_HISTORY_LIMIT)):
                    for item in listing:
                        ts = float(item.created_utc)
                        if not self._in_window(ts):
                            continue
                        if hasattr(item, "title"):
                            self._record_post(item, f"C_author_{cohort}")
                        else:
                            self._record_comment(item, f"C_author_{cohort}")
                self.store.mark_expanded(h)
            except prawcore.exceptions.NotFound:
                self.store.mark_expanded(h)     # suspended or deleted account
            except prawcore.exceptions.PrawcoreException as e:
                logging.warning("author expansion failed: %s", e)
                continue

            if i % 25 == 0:
                self.store.flush()
                logging.info("  %d/%d authors, %d items total", i, len(todo),
                             self.store.count())

        self.store.log("C", "done", cohort)

    # ---------------- Stage D: control cohort ----------------

    def stage_d_controls(self):
        """Authors from the same communities with no keyword hit, matched on
        tenure and volume. Without controls there is no discriminative claim,
        only a description of people who used certain phrases."""
        logging.info("STAGE D: control cohort")
        self.store.log("D", "start")

        n_target = int(self.store.author_count("seed") * self.cfg.CONTROL_RATIO)
        found = 0

        for sub_name in self.cfg.SUBREDDITS:
            sub = self.reddit.subreddit(sub_name)
            block = self.cfg.BLOCK_OF[sub_name]
            try:
                for p in sub.new(limit=None):
                    ts = float(p.created_utc)
                    if ts < self.cfg.WINDOW_START_TS:
                        break                   # .new() is reverse-chronological
                    if not self._in_window(ts):
                        continue
                    if not p.author:
                        continue
                    name = str(p.author)
                    h = self.pseudo.hash(name)
                    if not h:
                        continue
                    # skip anyone already seeded
                    row = self.store.conn.execute(
                        "SELECT 1 FROM authors WHERE author_hash=?", (h,)).fetchone()
                    if row:
                        continue
                    self._name_cache[h] = name
                    self.store.note_author(h, "control", block)
                    self._record_post(p, "D_control")
                    found += 1
                    if found >= n_target:
                        break
            except prawcore.exceptions.PrawcoreException as e:
                logging.warning("control sampling r/%s: %s", sub_name, e)
            self.store.flush()
            if found >= n_target:
                break

        logging.info("  %d control authors identified", found)
        self.store.log("D", "done", str(found))

    # ---------------- orchestration ----------------

    _name_cache: Dict[str, str] = {}

    def _cache_name(self, username: Optional[str]):
        if username:
            h = self.pseudo.hash(username)
            if h:
                self._name_cache[h] = username

    def run(self):
        t0 = time.time()
        try:
            self.stage_a_seed()
            self.stage_b_threads()
            self.stage_c_authors("seed")
            self.stage_d_controls()
            self.stage_c_authors("control")
        except KeyboardInterrupt:
            logging.info("interrupted; state is committed, rerun to continue")
        finally:
            self.store.refresh_author_stats()
            self.write_metadata(time.time() - t0)
            self.store.close()

    def write_metadata(self, elapsed: float):
        c = self.store.conn
        meta = {
            "collector_version": "6.0",
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "window": [self.cfg.WINDOW_START, self.cfg.WINDOW_END],
            "subreddits": {
                "anxiety_block": self.cfg.ANXIETY_BLOCK,
                "disengagement_block": self.cfg.DISENGAGEMENT_BLOCK,
            },
            "keywords": self.cfg.KEYWORDS,
            "keyword_role": "discovery only; not used as a label or feature",
            "filters_at_collection": "none (no score or length filtering)",
            "totals": {
                "items": c.execute("SELECT COUNT(*) FROM items").fetchone()[0],
                "posts": c.execute("SELECT COUNT(*) FROM items WHERE kind='post'").fetchone()[0],
                "comments": c.execute("SELECT COUNT(*) FROM items WHERE kind='comment'").fetchone()[0],
                "authors_seed": self.store.author_count("seed"),
                "authors_control": self.store.author_count("control"),
                "threads": c.execute("SELECT COUNT(*) FROM threads").fetchone()[0],
                "unexpanded_branches": c.execute(
                    "SELECT COALESCE(SUM(more_remaining),0) FROM threads").fetchone()[0],
                "authors_above_threshold": c.execute(
                    "SELECT COUNT(*) FROM authors WHERE n_items >= ?",
                    (self.cfg.MIN_RECORDS_PER_AUTHOR,)).fetchone()[0],
            },
            "by_stage": dict(c.execute(
                "SELECT stage, COUNT(*) FROM items GROUP BY stage").fetchall()),
        }
        with open(self.cfg.METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logging.info("metadata written to %s", self.cfg.METADATA_PATH)
        print(json.dumps(meta["totals"], indent=2))


# ==========================================================================

def main():
    for var in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"):
        if var not in os.environ:
            sys.exit(f"Set {var} as an environment variable. Do not hard-code credentials.")

    cfg = Config()
    print("=" * 66)
    print("CPST-FoMO Collector v6.0")
    print(f"  window      : {cfg.WINDOW_START} to {cfg.WINDOW_END}")
    print(f"  communities : {len(cfg.SUBREDDITS)} in 2 blocks")
    print(f"  keywords    : {len(cfg.FLAT_KEYWORDS)} (discovery only)")
    print(f"  storage     : {cfg.BASE_DIR}")
    print("=" * 66)

    Collector(cfg).run()


if __name__ == "__main__":
    main()
