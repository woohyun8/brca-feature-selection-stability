"""
Step 3 - Compare the gene lists.

Two kinds of comparison, for every panel size k:
  WITHIN-method : same method, different resamples      (all 50*49/2 = 1225 pairs of repeats)
  BETWEEN-method: different methods, SAME resample/split (one pair per repeat -> 50 values)

Overlap measures
  Jaccard   = |A and B| / |A or B|          0 = nothing shared, 1 = identical
  Kuncheva  = (r*n - k^2) / (k*(n-k))       Jaccard-like but corrected for chance overlap
              (r = shared genes, n = total genes, k = panel size)

95% CIs: bootstrap over the *repeats* (resample the 50 repeats with replacement, recompute the mean,
repeat 2000 times, take the 2.5th and 97.5th percentiles). Resampling repeats rather than pairs
avoids treating overlapping pairs as independent.

Outputs: results/within_method.csv, results/between_method.csv (means, CIs, AUROC summary).
"""
import argparse, pickle
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--results", default=str(ROOT / "results/results.pkl"))
parser.add_argument("--outdir", default=str(ROOT / "results"))
parser.add_argument("--n-genes", type=int, default=20530, help="total number of genes (for Kuncheva)")
parser.add_argument("--boot", type=int, default=2000)
args = parser.parse_args()

df = pd.DataFrame(pickle.load(open(args.results, "rb")))
METHODS = ["ttest", "l1_logistic", "random_forest"]
KS = sorted(df.panel_size.unique())
REPS = sorted(df.repeat.unique())
rng = np.random.RandomState(0)


def jaccard(a, b):
    return len(a & b) / len(a | b)


def kuncheva(a, b, k, n=args.n_genes):
    return (len(a & b) * n - k * k) / (k * (n - k))


def genes(method, k, rep):
    return df[(df.method == method) & (df.panel_size == k) & (df.repeat == rep)].genes.iloc[0]


def boot_ci(values, B=args.boot):
    values = np.asarray(values)
    means = [values[rng.randint(0, len(values), len(values))].mean() for _ in range(B)]
    return np.percentile(means, [2.5, 97.5])


def boot_ci_within(J, B=args.boot):
    """J = matrix of pairwise Jaccard between repeats. Resample repeat indices; average over pairs
    of two *different* original repeats."""
    n = len(J)
    means = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        sub = J[np.ix_(idx, idx)]
        mask = idx[:, None] != idx[None, :]
        means.append(sub[mask].mean())
    return np.percentile(means, [2.5, 97.5])


# ---------- within-method ----------
within = []
for m in METHODS:
    for k in KS:
        panels = [genes(m, k, r) for r in REPS]
        n = len(panels)
        J = np.eye(n); K = np.eye(n)
        for i, j in combinations(range(n), 2):
            J[i, j] = J[j, i] = jaccard(panels[i], panels[j])
            K[i, j] = K[j, i] = kuncheva(panels[i], panels[j], k)
        off = ~np.eye(n, dtype=bool)
        lo, hi = boot_ci_within(J)
        au = df[(df.method == m) & (df.panel_size == k)].auroc
        within.append({"method": m, "panel_size": k, "mean_jaccard": J[off].mean(), "ci_low": lo, "ci_high": hi,
                       "mean_kuncheva": K[off].mean(), "mean_auroc": au.mean(), "min_auroc": au.min()})

# ---------- between-method (same split) ----------
between = []
for k in KS:
    for m1, m2 in combinations(METHODS, 2):
        jv = [jaccard(genes(m1, k, r), genes(m2, k, r)) for r in REPS]
        kv = [kuncheva(genes(m1, k, r), genes(m2, k, r), k) for r in REPS]
        lo, hi = boot_ci(jv)
        between.append({"pair": f"{m1}_vs_{m2}", "panel_size": k, "mean_jaccard": np.mean(jv),
                        "ci_low": lo, "ci_high": hi, "mean_kuncheva": np.mean(kv)})

outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
w, b = pd.DataFrame(within), pd.DataFrame(between)
w.to_csv(outdir / "within_method.csv", index=False)
b.to_csv(outdir / "between_method.csv", index=False)
pd.set_option("display.width", 200)
print("=== WITHIN-method (same method, different resamples) ===")
print(w.round(3).to_string(index=False))
print("\n=== BETWEEN-method (same split, different methods) ===")
print(b.round(3).to_string(index=False))
print(f"\nWithin Jaccard range : {w.mean_jaccard.min():.2f} - {w.mean_jaccard.max():.2f}")
print(f"Between Jaccard range: {b.mean_jaccard.min():.3f} - {b.mean_jaccard.max():.3f}  (max CI upper bound {b.ci_high.max():.3f})")
print(f"AUROC: mean >= {w.mean_auroc.min():.4f}, worst single run {w.min_auroc.min():.4f}")
