#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The three figures the manuscript references but does not yet have
=================================================================

  fig1_operationalisation   which data channel feeds which CPST space, and
                            which edges are inference rather than observation.
                            This is the figure that makes the paper's central
                            asymmetry visible before any result.

  fig2_cohort_flow          CONSORT-style construction of the analysis sample,
                            with counts at every stage. Answers the
                            reproducibility questions a reviewer will ask about
                            Section IV.

  fig3_offset_diagnostic    one author's activity over 24 hours UTC, the
                            detected trough, and the same activity realigned to
                            estimated local time. Shows the estimator working on
                            a real case rather than asserting that it does.

Also writes table1_corpus.tex with the cohort medians filled in, which the
manuscript currently carries as placeholders.

Usage
    python make_figures.py
    python make_figures.py --author <author_hash>     # pick the exemplar
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Data folder: set CPST_BASE_DIR to override the default location.
BASE_DIR = os.environ.get(
    "CPST_BASE_DIR",
    os.path.join(os.path.expanduser("~"), "Documents", "CPST_FOMO_v6"))
DB_PATH = os.path.join(BASE_DIR, "cpst_fomo.db")
FEATURES = os.path.join(BASE_DIR, "cpst_features.csv")
OUT_DIR = os.path.join(BASE_DIR, "paper1")

WINDOW_START_TS = datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()
WINDOW_END_TS = datetime(2025, 12, 31, tzinfo=timezone.utc).timestamp()

BLOCK_COLOR = {"C": "#2a78d6", "P": "#eb6834", "S": "#1baf7a", "T": "#4a3aa7"}
BLOCK_NAMES = {"C": "Cyber", "P": "Physical", "S": "Social", "T": "Thinking"}
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE, GRID, PANEL = "#ffffff", "#e6e5e1", "#f7f7f5"
COL_1, COL_2 = 3.5, 7.16


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 8,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "axes.edgecolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.labelcolor": INK,
        "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"),
                    dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def box(ax, x, y, w, h, text, fc, ec=None, fontsize=7, weight="normal",
        tc=None, align="center"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.8, edgecolor=ec or fc, facecolor=fc, zorder=2))
    ax.text(x + (w / 2 if align == "center" else 0.014), y + h / 2, text,
            ha=align, va="center", fontsize=fontsize, color=tc or INK,
            zorder=3, fontweight=weight, linespacing=1.45)


def arrow(ax, p0, p1, color, dashed=False, lw=1.1):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=7, linewidth=lw,
        color=color, zorder=1,
        linestyle=(0, (2.2, 1.8)) if dashed else "solid",
        shrinkA=1, shrinkB=2))


# ======================================================================
# Figure 1: operationalisation
# ======================================================================

def fig1_operationalisation():
    fig, ax = plt.subplots(figsize=(COL_2, 2.9))
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")

    channels = [
        ("Community &\nplatform metadata", 0.845),
        ("Record\ntimestamps", 0.615),
        ("Thread structure\n(parent, depth)", 0.385),
        ("Record\ntext", 0.155),
    ]
    for label, y in channels:
        box(ax, 0.015, y - 0.075, 0.205, 0.15, label, PANEL, ec="#d8d7d2",
            fontsize=7, tc=INK_2)

    spaces = [
        ("C", "Cyber\ncommunity portfolio, entropy,\nchurn, reception, link domains", 0.845),
        ("P", "Physical\ncircadian position, burstiness,\nsessions, activity density", 0.615),
        ("S", "Social\nreply structure, reciprocity,\npartners, thread position", 0.385),
        ("T", "Thinking\ncomparison, compulsion, affect,\nsomatic and sleep vocabulary", 0.155),
    ]
    for key, label, y in spaces:
        c = BLOCK_COLOR[key]
        box(ax, 0.50, y - 0.085, 0.36, 0.17, label, c + "1f", ec=c,
            fontsize=7, align="left")
        ax.add_patch(plt.Rectangle((0.50, y - 0.085), 0.0085, 0.17,
                                   color=c, zorder=4, lw=0))

    # channel -> space edges. dashed = inferred rather than observed
    edges = [
        (0.845, 0.845, "C", False),
        (0.845, 0.385, "S", False),
        (0.615, 0.615, "P", True),
        (0.615, 0.845, "C", False),
        (0.385, 0.385, "S", False),
        (0.385, 0.615, "P", True),
        (0.155, 0.155, "T", False),
    ]
    for y0, y1, key, dashed in edges:
        arrow(ax, (0.222, y0), (0.498, y1), BLOCK_COLOR[key], dashed=dashed)
    # Sleep vocabulary is a THINKING-space feature. It touches the physical
    # space only as the independent channel used to validate the offset, so
    # it is drawn as a grey dotted check, not as a feature edge.
    arrow(ax, (0.222, 0.155), (0.498, 0.615), INK_MUTED, dashed=True, lw=0.8)
    ax.text(0.318, 0.262, "validation only", fontsize=6, color=INK_MUTED,
            style="italic", ha="center", va="center", rotation=34)

    ax.text(0.375, 0.985, "observed", ha="left", fontsize=6.8, color=INK_2,
            va="center")
    ax.plot([0.300, 0.360], [0.985, 0.985], color=INK_2, lw=1.1)
    ax.text(0.375, 0.912, "inferred", ha="left", fontsize=6.8, color=INK_2,
            va="center")
    ax.plot([0.300, 0.360], [0.912, 0.912], color=INK_2, lw=1.1,
            linestyle=(0, (2.2, 1.8)))

    ax.text(0.885, 0.615,
            "no sensed\nmeasurement",
            ha="left", va="center", fontsize=6.8, color=BLOCK_COLOR["P"],
            style="italic")

    ax.text(0.015, 0.985, "Data channel", fontsize=7.5, color=INK,
            fontweight="bold")
    ax.text(0.50, 0.985, "CPST space", fontsize=7.5, color=INK,
            fontweight="bold")
    save(fig, "fig1_operationalisation")


# ======================================================================
# Figure 2: cohort flow
# ======================================================================

# --- cohort flow: card layout with drawn icons --------------------------
from matplotlib.patches import Circle, Rectangle, Ellipse, Polygon, Wedge

FL_INK, FL_MUTED = "#1d2733", "#5f6b7a"
BLUE1, BLUE2, BLUE3 = "#4f8fea", "#2f6fd1", "#1f4fa0"
NAVY = "#17375e"
ROSE, ROSE_BG = "#c8394a", "#fdeef0"
SEED, CTRL = "#e8772e", "#12968f"
VIOL, VIOL_BG = "#6b4fc8", "#f1edfc"
FL_CARD_BG = "#ffffff"
FL_PAGE = "#ffffff"

def tint(hexc, a):
    c = np.array(matplotlib.colors.to_rgb(hexc))
    return tuple(1 - a * (1 - c))

# ---------------- icons (drawn white, centred at x,y, size r) -------------
def icon_database(ax, x, y, r, c="white"):
    from matplotlib.patches import Arc
    w, h, e = r * 1.15, r * 1.25, r * 0.36
    ax.add_patch(Rectangle((x - w/2, y - h/2), w, h, fc=c, ec="none", zorder=6))
    ax.add_patch(Ellipse((x, y - h/2), w, e, fc=c, ec="none", zorder=6))
    ax.add_patch(Ellipse((x, y + h/2), w, e, fc=c, ec=ACC, lw=0.6, zorder=7))
    for yy in (y + h/6, y - h/6):
        ax.add_patch(Arc((x, yy), w, e, theta1=180, theta2=360, ec=ACC, lw=0.7, zorder=7))

def icon_calendar(ax, x, y, r, c="white"):
    w, h = r * 1.25, r * 1.1
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0,rounding_size=0.025",
                                fc=c, ec="none", zorder=6))
    ax.add_patch(Rectangle((x - w/2, y + h/2 - h*0.28), w, h*0.28, fc=ACC, ec="none", alpha=0.35, zorder=7))
    for i in range(3):
        for j in range(2):
            ax.add_patch(Rectangle((x - w*0.33 + i*w*0.26, y - h*0.32 + j*h*0.26),
                                   w*0.14, h*0.14, fc=ACC, ec="none", zorder=7))
    for dx in (-w*0.25, w*0.25):
        ax.add_patch(Rectangle((x + dx - 0.012, y + h/2 - 0.01), 0.024, h*0.2, fc=c, ec="none", zorder=7))

def icon_funnel(ax, x, y, r, c="white"):
    w, h = r * 1.4, r * 1.2
    pts = [(x - w/2, y + h/2), (x + w/2, y + h/2), (x + w*0.1, y - h*0.05),
           (x + w*0.1, y - h/2), (x - w*0.1, y - h*0.35), (x - w*0.1, y - h*0.05)]
    ax.add_patch(Polygon(pts, closed=True, fc=c, ec="none", zorder=6))

def icon_people(ax, x, y, r, c="white"):
    for dx, s in ((-r*0.38, 0.8), (r*0.38, 0.8), (0, 1.0)):
        ax.add_patch(Circle((x + dx, y + r*0.28*s), r*0.22*s, fc=c, ec=ACC if s == 1 else "none",
                            lw=0.6, zorder=7 if s == 1 else 6))
        ax.add_patch(Wedge((x + dx, y - r*0.42*s), r*0.4*s, 0, 180, fc=c,
                           ec=ACC if s == 1 else "none", lw=0.6, zorder=7 if s == 1 else 6))

def icon_link(ax, x, y, r, c="white"):
    for dx in (-r*0.27, r*0.27):
        ax.add_patch(Circle((x + dx, y), r*0.38, fc="none", ec=c, lw=1.6, zorder=6))

def icon_moon(ax, x, y, r, c="white", bg=None):
    ax.add_patch(Circle((x, y), r*0.55, fc=c, ec="none", zorder=6))
    ax.add_patch(Circle((x + r*0.28, y + r*0.2), r*0.48, fc=bg, ec="none", zorder=7))
    for (dx, dy, s) in ((-r*0.05, r*0.62, 0.07), (r*0.55, -r*0.35, 0.055)):
        ax.add_patch(Circle((x + dx, y + dy), r*s, fc=c, ec="none", zorder=7))

ACC = BLUE1

def card(ax, x, y, w, h, accent, bg=FL_CARD_BG, lw=0.9):
    ax.add_patch(FancyBboxPatch((x+0.012, y-0.018), w, h,
        boxstyle="round,pad=0,rounding_size=0.06", fc="#000000", ec="none", alpha=0.07, zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
        fc=bg, ec=tint(accent, 0.55), lw=lw, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y), 0.055, h, boxstyle="round,pad=0,rounding_size=0.027",
        fc=accent, ec="none", zorder=3))

def badge(ax, x, y, r, accent, draw, **kw):
    global ACC
    ACC = accent
    ax.add_patch(Circle((x, y), r, fc=accent, ec="white", lw=1.2, zorder=5))
    draw(ax, x, y, r*0.62, **kw)

def down_arrow(ax, x, y0, y1, color):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18",
                                color=color, lw=1.4, shrinkA=0, shrinkB=0), zorder=4)

def _flow_figure(stats):
    W, H = 3.5, 4.10
    fig = plt.figure(figsize=(W, H), facecolor=FL_PAGE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.set_aspect("equal")

    X, CW, CH, GAP = 0.06, 2.12, 0.60, 0.235
    BR = 0.17                    # badge radius
    xs_badge = X + 0.30
    xt = X + 0.56                # text start
    tops = [H - 0.06 - i*(CH + GAP) for i in range(5)]

    def stage_text(y, title, big, small, col):
        ax.text(xt, y + CH*0.73, title, fontsize=6.2, color=col, fontweight="bold",
                va="center", ha="left", zorder=8)
        ax.text(xt, y + CH*0.43, big, fontsize=8.2, color=FL_INK, fontweight="bold",
                va="center", ha="left", zorder=8)
        ax.text(xt, y + CH*0.17, small, fontsize=5.6, color=FL_MUTED, va="center", ha="left", zorder=8)

    # 1 collected
    y = tops[0] - CH
    card(ax, X, y, CW, CH, BLUE1)
    badge(ax, xs_badge, y + CH/2, BR, BLUE1, icon_database)
    stage_text(y, "1  COLLECTED", f"{stats['records']:,} records",
               f"{stats['authors']:,} authors, four collection stages", BLUE1)
    # 2 in window
    y2 = tops[1] - CH
    card(ax, X, y2, CW, CH, BLUE2)
    badge(ax, xs_badge, y2 + CH/2, BR, BLUE2, icon_calendar)
    stage_text(y2, "2  IN WINDOW", f"{stats['in_window']:,} records",
               f"{stats['in_window_authors']:,} authors, 2023\u20132025", BLUE2)
    # 3 analysis sample
    y3 = tops[2] - CH
    card(ax, X, y3, CW, CH, BLUE3)
    badge(ax, xs_badge, y3 + CH/2, BR, BLUE3, icon_funnel)
    stage_text(y3, "3  ANALYSIS SAMPLE", f"{stats['author_windows']:,} authors",
               f"{stats['retained']:,} records, 30+ each", BLUE3)
    # 4 cohorts split
    y4 = tops[3] - CH
    card(ax, X, y4, CW, CH, NAVY)
    badge(ax, xs_badge, y4 + CH/2, BR, NAVY, icon_people)
    ax.text(xt, y4 + CH*0.84, "4  COHORTS", fontsize=6.2, color=NAVY, fontweight="bold",
            va="center", zorder=8)
    pw, ph = 0.70, 0.37
    for i, (lab, n, col) in enumerate((("seed", stats["seed"], SEED),
                                        ("control", stats["control"], CTRL))):
        px = xt + i*(pw + 0.08); py = y4 + 0.06
        ax.add_patch(FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=tint(col, 0.14), ec=col, lw=0.8, zorder=6))
        ax.text(px + pw/2, py + ph*0.63, f"{n:,}", fontsize=7.8, fontweight="bold",
                color=col, va="center", ha="center", zorder=8)
        ax.text(px + pw/2, py + ph*0.24, lab, fontsize=5.4, color=col, va="center",
                ha="center", zorder=8)
    # 5 matched
    y5 = tops[4] - CH
    card(ax, X, y5, CW, CH, NAVY, bg=tint(NAVY, 0.05))
    badge(ax, xs_badge, y5 + CH/2, BR, NAVY, icon_link)
    unmatched = stats["author_windows"] - 2*stats["pairs"]
    stage_text(y5, "5  MATCHED PAIRS",
               f"{stats['pairs']:,} pairs",
               f"on volume + tenure, {stats['pair_pct']:.0f}% of seed", NAVY)

    cx = X + CW/2 + 0.25
    for (a, b, col) in ((y, tops[1], BLUE1), (y2, tops[2], BLUE2),
                        (y3, tops[3], BLUE3), (y4, tops[4], NAVY)):
        down_arrow(ax, xs_badge, a, b, col)

    # ---------------- exclusions (right) ----------------
    RX, RW = X + CW + 0.13, W - (X + CW + 0.13) - 0.05
    excl = [
        ((y + tops[1]) / 2, f"\u2212{stats['records']-stats['in_window']:,} records",
         f"\u2212{stats['authors']-stats['in_window_authors']:,} authors",
         "outside window or\nparticipant cohort"),
        ((y2 + tops[2]) / 2, f"\u2212{stats['in_window_authors']-stats['author_windows']:,} authors",
         f"\u2212{stats['in_window']-stats['retained']:,} records",
         "fewer than 30\nrecords in window"),
    ]
    for yc, l1, l2, why in excl:
        eh = 0.56
        ax.add_patch(FancyBboxPatch((RX, yc - eh/2), RW, eh,
            boxstyle="round,pad=0,rounding_size=0.05", fc=ROSE_BG, ec=tint(ROSE, 0.5),
            lw=0.7, zorder=2))
        ax.plot([xs_badge + 0.03, RX], [yc, yc], color=tint(ROSE, 0.6), lw=0.8,
                ls=(0, (2, 1.5)), zorder=1)
        ax.add_patch(Circle((xs_badge, yc), 0.028, fc=ROSE, ec="white", lw=0.6, zorder=5))
        ax.text(RX + 0.07, yc + 0.165, l1, fontsize=5.9, color=ROSE, fontweight="bold",
                va="center", zorder=8)
        ax.text(RX + 0.07, yc + 0.055, l2, fontsize=5.3, color=ROSE, va="center", zorder=8)
        ax.text(RX + 0.07, yc - 0.13, why, fontsize=5.0, color=FL_MUTED, va="center",
                linespacing=1.1, zorder=8)

    # ---------------- validation branch ----------------
    vh = 1.05
    vy = y5 + 0.0
    card(ax, RX, vy, RW, vh + (y4 - y5) - 0.0 - 0.0 if False else vh + 0.40, VIOL, bg=VIOL_BG)
    vtop = vy + vh + 0.40
    badge(ax, RX + RW/2 + 0.02, vtop - 0.25, 0.15, VIOL, icon_moon, bg=VIOL)
    ax.text(RX + RW/2 + 0.02, vtop - 0.52, "CIRCADIAN\nVALIDATION", fontsize=5.6,
            color=VIOL, fontweight="bold", ha="center", va="center", linespacing=1.05, zorder=8)
    ax.text(RX + RW/2 + 0.02, vtop - 0.82, f"{stats['validation_n']:,}", fontsize=9,
            color=FL_INK, fontweight="bold", ha="center", va="center", zorder=8)
    ax.text(RX + RW/2 + 0.02, vtop - 0.98, "authors", fontsize=5.4, color=FL_MUTED,
            ha="center", va="center", zorder=8)
    ax.text(RX + RW/2 + 0.02, vy + 0.22, "trusted offset,\n10+ records by\nnight and by day",
            fontsize=4.9, color=FL_MUTED, ha="center", va="center", linespacing=1.1, zorder=8)
    # branch from stage 3
    bx = RX + RW/2 + 0.02
    ax.plot([X + CW, bx, bx], [y3 + CH/2, y3 + CH/2, vtop + 0.04], color=VIOL, lw=1.0,
            ls=(0, (3, 1.6)), zorder=1, solid_capstyle="round")
    ax.annotate("", xy=(bx, vtop + 0.005), xytext=(bx, vtop + 0.08),
                arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.15",
                                color=VIOL, lw=1.0), zorder=4)
    ax.text(X + CW + 0.06, y3 + CH/2 + 0.07, "drawn from all", fontsize=4.8, color=VIOL,
            style="italic", va="bottom", zorder=8)

    return fig



def fig2_cohort_flow(stats: dict):
    """Cohort flow drawn as stage cards with icons, red exclusion callouts and
    a purple branch for the circadian validation subsample."""
    stats = dict(stats)
    if not stats.get("in_window_authors"):
        sys.exit("Fig. 2 needs the in-window author count; check the database path.")
    if not stats.get("validation_n"):
        sys.exit("Fig. 2 needs mh_robustness.json; run mh_robustness.py first.")
    fig = _flow_figure(stats)
    save(fig, "fig2_cohort_flow")


# ======================================================================
# Figure 3: offset diagnostic
# ======================================================================

ANCHOR_LOCAL_HOUR = 4   # must be an INTEGER; see the note below


def estimate_offset(counts: np.ndarray, anchor: int = ANCHOR_LOCAL_HOUR):
    """Recover an author's UTC offset from their six-hour activity trough.

    The anchor must be an INTEGER and must match ANCHOR_LOCAL_HOUR in
    extract_cpst_features.py. With a half-integer anchor such as 03:30,
    (anchor - trough_centre) always ends in .5 because trough_centre is an
    integer, and Python rounds halves to even, so only EVEN offsets can ever
    be produced and every odd time zone is folded into a neighbour.
    """
    wrapped = np.concatenate([counts, counts[:6]])
    win = np.array([wrapped[i:i + 6].sum() for i in range(24)])
    trough_start = int(np.argmin(win))
    trough_centre = (trough_start + 3) % 24
    off = int(round((anchor - trough_centre) % 24))
    if off > 12:
        off -= 24
    depth = float(win[trough_start] / (counts.sum() / 4.0))
    return off, depth, trough_start


def fig3_offset_diagnostic(db: str, feat: pd.DataFrame, author: str | None):
    if author is None:
        cand = feat[(feat.P_rhythm_trough_depth < 0.35) & (feat.n_items > 300)]
        if cand.empty:
            cand = feat.sort_values("P_rhythm_trough_depth").head(20)
        author = cand.sort_values("n_items", ascending=False).iloc[0].author_hash
    print(f"  exemplar author: {author}")

    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT created_utc FROM items WHERE author_hash=? "
        "AND created_utc BETWEEN ? AND ?",
        (author, WINDOW_START_TS, WINDOW_END_TS)).fetchall()
    conn.close()
    ts = np.array([r[0] for r in rows], float)
    if len(ts) == 0:
        print("  no records for that author, skipping figure 3")
        return
    hours = ((ts // 3600) % 24).astype(int)
    counts = np.bincount(hours, minlength=24).astype(float)
    off, depth, tstart = estimate_offset(counts)
    print(f"  n={len(ts):,}  estimated offset UTC{off:+d}  trough depth {depth:.3f}")

    local = (hours + off) % 24
    lcounts = np.bincount(local, minlength=24).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(COL_2, 2.0), sharey=True)

    ax = axes[0]
    ax.bar(np.arange(24), counts, width=0.82, color=INK_MUTED, lw=0)
    for h in range(tstart, tstart + 6):
        ax.bar(h % 24, counts[h % 24], width=0.82,
               color=BLOCK_COLOR["P"], lw=0)
    ax.set_xlabel("hour, UTC")
    ax.set_ylabel("records")
    ax.set_xticks(range(0, 24, 4))
    # trough window goes in the title so it cannot collide with a tall bar
    ax.set_title(f"Observed, $n$ = {len(ts):,}; trough "
                 f"{tstart:02d}:00–{(tstart+6)%24:02d}:00 UTC", loc="left")

    ax = axes[1]
    ax.bar(np.arange(24), lcounts, width=0.82, color=INK_MUTED, lw=0)
    for h in range(0, 5):
        ax.bar(h, lcounts[h], width=0.82, color=BLOCK_COLOR["P"], lw=0)
    ax.axvline(ANCHOR_LOCAL_HOUR, color=INK_2, lw=1.2, ls=":")
    ax.text(ANCHOR_LOCAL_HOUR + 0.25, ax.get_ylim()[1] * 0.93,
            f"{ANCHOR_LOCAL_HOUR:02d}:00 anchor", fontsize=6.5,
            color=INK_2, va="top")
    ax.set_xlabel("hour, estimated local time")
    ax.set_xticks(range(0, 24, 4))
    ax.set_title(f"Realigned, offset UTC{off:+d}, trough depth {depth:.2f}",
                 loc="left")

    for a in axes:
        a.grid(axis="y", lw=0.5)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    fig.tight_layout()
    save(fig, "fig3_offset_diagnostic")


# ======================================================================
# Table I
# ======================================================================

def write_table1(feat: pd.DataFrame, stats: dict):
    rows = []
    for cohort in ("seed", "control"):
        s = feat[feat.cohort == cohort]
        if s.empty:
            continue
        rows.append(
            f"{cohort.capitalize()} & {len(s):,} & {s.n_items.median():.0f} & "
            f"{s.tenure_days.median():.0f} & "
            f"{s.C_n_communities.median():.0f} & "
            f"{s.P_active_days.median():.0f} & "
            f"{s.S_n_partners.median():.0f} \\\\")
    tex = r"""\begin{table}[!t]
\caption{Corpus description by cohort. Values are medians.}
\label{tab:corpus}
\centering
\begin{tabular}{lrrrrrr}
\toprule
Cohort & $n$ & Records & Tenure & Comm. & Active & Partners \\
 & & & (days) & & days & \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
\multicolumn{7}{l}{Matched pairs: """ + f"{stats['pairs']:,}" + \
r""" within a 0.5\,SD caliper, """ + f"{stats['pair_pct']:.0f}" + \
r"""\% of seed authors} \\
\bottomrule
\end{tabular}
\end{table}
"""
    p = os.path.join(OUT_DIR, "table1_corpus.tex")
    open(p, "w", encoding="utf-8").write(tex)
    print(f"  wrote table1_corpus.tex")
    for r in rows:
        print("    " + r.replace(r"\\", "").strip())


# ======================================================================

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


def count_in_window_authors(db: str):
    """Distinct seed and control authors with at least one in-window record,
    using the same join and window as the feature extractor."""
    if not os.path.exists(db):
        return None
    conn = sqlite3.connect(db)
    try:
        n = conn.execute("""
            SELECT COUNT(DISTINCT i.author_hash)
            FROM items i JOIN authors a ON a.author_hash = i.author_hash
            WHERE i.author_hash IS NOT NULL
              AND a.cohort IN ('seed','control')
              AND i.created_utc BETWEEN ? AND ?
        """, (WINDOW_START_TS, WINDOW_END_TS)).fetchone()[0]
    except sqlite3.Error as e:
        print(f"  could not count in-window authors: {e}")
        n = None
    conn.close()
    if n is not None:
        print(f"  in-window seed+control authors: {n:,}")
    return n


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--features", default=FEATURES)
    ap.add_argument("--author", default=None)
    ap.add_argument("--in-window-authors", type=int, default=None,
                    help="override; by default counted from the database")
    args = _args(ap)

    setup_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    feat = pd.read_csv(args.features) if os.path.exists(args.features) else pd.DataFrame()
    if feat.empty:
        sys.exit(f"Feature file not found: {args.features}")

    n_seed = int((feat.cohort == "seed").sum())
    n_ctrl = int((feat.cohort == "control").sum())
    pairs = int(feat.matched.sum() // 2) if "matched" in feat.columns else 0
    stats = {
        "records": 982276, "authors": 12823,
        "in_window": 952556, "retained": 935329,
        "author_windows": len(feat), "seed": n_seed, "control": n_ctrl,
        "pairs": pairs,
        "pair_pct": 100 * pairs / max(n_seed, 1),
        "in_window_authors": args.in_window_authors or
                             count_in_window_authors(args.db),
        "validation_n": None,
    }
    mh_json = os.path.join(OUT_DIR, "mh_robustness.json")
    if os.path.exists(mh_json):
        import json
        res = json.load(open(mh_json, encoding="utf-8"))["results"]
        head = [r for r in res if r["anchor"] == 4.0 and r["night"] == [0, 5]]
        if head:
            stats["validation_n"] = head[0]["n_authors"]
    else:
        print("  run mh_robustness.py first to add the validation box to Fig. 2")

    print("figures")
    fig1_operationalisation()
    fig2_cohort_flow(stats)
    if os.path.exists(args.db):
        fig3_offset_diagnostic(args.db, feat, args.author)
    else:
        print("  database not found, skipping figure 3")

    print("tables")
    write_table1(feat, stats)
    print(f"\noutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
