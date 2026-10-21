"""
Step 2 - The main experiment.

Repeat N times:
  1. Split PATIENTS (not samples) into 80% train / 20% test.
  2. On the train part only, rank all genes with three feature-selection methods:
       - t-test             : genes whose tumor/normal means differ most
       - L1 logistic        : genes kept by a sparse (L1-penalised) logistic regression
       - random forest      : genes the forest uses most
  3. For each panel size k in 5/10/25/50/100: take the top-k genes of each method, train the SAME
     logistic regression on them, and measure AUROC on the held-out test patients.
  4. Store the chosen gene set. (Step 3 compares these gene sets.)

Why split by patient? A few patients contribute both a tumor and a normal sample; if one went
into train and the other into test, information would leak.

L1 detail: an L1 model with a fixed penalty keeps only ~30 genes, so it cannot fill a top-50 or
top-100 panel. For each k we therefore use the smallest C (weakest penalty step) on a grid for which
the L1 model has at least k non-zero coefficients, then take the k largest |coefficients|.
"""
import argparse, pickle, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", module="sklearn")   # silence harmless deprecation warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)   # scipy t-test precision notes on near-constant genes

ROOT = Path(__file__).resolve().parents[1]
PANEL_SIZES = [5, 10, 25, 50, 100]
C_GRID = np.geomspace(0.02, 5.0, 16)       # L1 penalty strengths to try (smaller C = stronger penalty)

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(ROOT / "data/processed"))
parser.add_argument("--out", default=str(ROOT / "results/results.pkl"))
parser.add_argument("--repeats", type=int, default=50, help="number of resamples (use 3 for a quick test)")
parser.add_argument("--train-size", type=float, default=0.8, help="fraction of patients used for training")
args = parser.parse_args()

X = pd.read_pickle(Path(args.data) / "X_expr.pkl")
y = pd.read_pickle(Path(args.data) / "y_labels.pkl")
patient_ids = pd.read_pickle(Path(args.data) / "patient_ids.pkl")
gene_names = X.columns.values
X_np, y_np, patients_np = X.values.astype(np.float32), y.values, patient_ids.values

# One row per patient; pseudo-label = "this patient has a normal sample". Used only so that
# the (few) patients with normals are spread evenly over train and test.
pdf = pd.DataFrame({"patient": patients_np, "label": y_np})
has_normal = pdf.groupby("patient")["label"].apply(lambda s: int((s == 0).any()))
unique_patients, pseudo = has_normal.index.values, has_normal.values
print(f"{X.shape[0]} samples, {len(unique_patients)} patients ({pseudo.sum()} with a normal sample)")

splitter = StratifiedShuffleSplit(n_splits=50, train_size=args.train_size, random_state=42)
results, t0 = [], time.time()

for rep, (tr_idx, te_idx) in enumerate(splitter.split(unique_patients, pseudo)):
    if rep >= args.repeats:
        break
    train_p, test_p = set(unique_patients[tr_idx]), set(unique_patients[te_idx])
    tr = np.array([p in train_p for p in patients_np])
    te = np.array([p in test_p for p in patients_np])

    scaler = StandardScaler()                       # fit on TRAIN only -> no leakage
    Xtr, Xte = scaler.fit_transform(X_np[tr]), scaler.transform(X_np[te])
    ytr, yte = y_np[tr], y_np[te]

    # ---- ranking 1: t-test (rank genes by |t|) ----
    t_stat, _ = stats.ttest_ind(Xtr[ytr == 1], Xtr[ytr == 0], axis=0, equal_var=False)
    rank_ttest = np.argsort(-np.abs(np.nan_to_num(t_stat, nan=0.0)))

    # ---- ranking 3: random forest importance ----
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=0).fit(Xtr, ytr)
    rank_rf = np.argsort(-rf.feature_importances_)

    # ---- ranking 2: L1 logistic, exact-k procedure (see docstring) ----
    l1_coef_for_k = {}
    for C in C_GRID:
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000, random_state=0).fit(Xtr, ytr)
        coef = np.abs(m.coef_.ravel())
        nonzero = int((coef > 0).sum())
        for k in PANEL_SIZES:
            if k not in l1_coef_for_k and nonzero >= k:
                l1_coef_for_k[k] = (C, coef)
        if len(l1_coef_for_k) == len(PANEL_SIZES):
            break
    assert len(l1_coef_for_k) == len(PANEL_SIZES), "C grid too small to reach the largest panel size"

    for k in PANEL_SIZES:
        top_sets = {
            "ttest": rank_ttest[:k],
            "random_forest": rank_rf[:k],
            "l1_logistic": np.argsort(-l1_coef_for_k[k][1])[:k],
        }
        assert (l1_coef_for_k[k][1][top_sets["l1_logistic"]] > 0).all()   # no zero-coefficient padding
        for method, top in top_sets.items():
            clf = LogisticRegression(max_iter=1000, random_state=0).fit(Xtr[:, top], ytr)
            prob = clf.predict_proba(Xte[:, top])[:, 1]
            results.append({"repeat": rep, "method": method, "panel_size": k,
                            "auroc": roc_auc_score(yte, prob), "auprc": average_precision_score(yte, prob),
                            "genes": frozenset(gene_names[top])})
    print(f"repeat {rep + 1}/{args.repeats} done ({time.time() - t0:.0f}s)", flush=True)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "wb") as f:
    pickle.dump(results, f)
print("Saved", len(results), "rows to", args.out)
