"""Step 4 - Figures: (1) AUROC vs panel size, (2) within- vs between-method Jaccard with 95% CI bands."""
import argparse, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--results", default=str(ROOT / "results"))
parser.add_argument("--figdir", default=str(ROOT / "figs"))
args = parser.parse_args()
res = Path(args.results); figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#1a1a1a", "#6b6b6b", "#e5e5e5"
plt.rcParams.update({"font.size": 17, "axes.edgecolor": MUTED})
w, b = pd.read_csv(res / "within_method.csv"), pd.read_csv(res / "between_method.csv")
raw = pd.DataFrame(pickle.load(open(res / "results.pkl", "rb")))
KS = [5, 10, 25, 50, 100]
NAMES = {"ttest": "t-test", "l1_logistic": "L1-logistic", "random_forest": "Random forest"}
COL = {"ttest": BLUE, "l1_logistic": ORANGE, "random_forest": AQUA}


def style(ax):
    ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS]); ax.minorticks_off()
    ax.grid(True, color=GRID); ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED); ax.set_xlabel("Gene panel size (number of genes)", color=INK)


def band(ax, sub, color, label, marker):
    ax.fill_between(sub.panel_size, sub.ci_low, sub.ci_high, color=color, alpha=.2, lw=0)
    ax.plot(sub.panel_size, sub.mean_jaccard, marker=marker, color=color, lw=2.5, ms=7, label=label)


# Figure 1: accuracy
fig, ax = plt.subplots(figsize=(7, 3.5))
for m in NAMES:
    ax.plot(KS, raw[raw.method == m].groupby("panel_size").auroc.mean(), marker="o", color=COL[m], lw=3, ms=9, label=NAMES[m])
ax.set_ylim(0.995, 1.0005); ax.set_ylabel("Mean test AUROC", color=INK); style(ax)
ax.legend(frameon=False, loc="lower right"); fig.tight_layout()
fig.savefig(figdir / "fig1_auroc.png", dpi=200, facecolor="white")

# Figure 2: within vs between
fig, axs = plt.subplots(2, 1, figsize=(8, 7.6), sharey=True)
for m in NAMES:
    band(axs[0], w[w.method == m], COL[m], NAMES[m], "o")
for pair, c, lab in [("ttest_vs_l1_logistic", BLUE, "t-test vs L1"), ("ttest_vs_random_forest", ORANGE, "t-test vs RF"),
                     ("l1_logistic_vs_random_forest", AQUA, "L1 vs RF")]:
    band(axs[1], b[b.pair == pair], c, lab, "s")
axs[0].set_title("Same method, different resamples", loc="left", color=INK)
axs[1].set_title("Same split, different methods", loc="left", color=INK)
axs[0].set_ylim(0, 1.18); axs[0].set_yticks([0, .2, .4, .6, .8, 1.0])
fig.supylabel("Mean Jaccard overlap (95% CI)", color=INK, fontsize=17)
for a in axs:
    style(a); a.legend(frameon=False, loc="upper center", ncol=3, fontsize=15, columnspacing=1, handlelength=1.4)
fig.tight_layout(); fig.savefig(figdir / "fig2_within_vs_between.png", dpi=200, facecolor="white")
print("Saved figures to", figdir)
