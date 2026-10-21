"""
Step 5 - Which genes do the methods actually pick?

For a chosen panel size k, count in how many of the 50 repeats each gene was selected
("selection frequency") and show the most frequently selected genes of each method side by side.
Also lists genes that two or more methods select often (consensus candidates).

Output: results/top_genes_k{k}.csv and a printed summary.
"""
import argparse, pickle
from collections import Counter
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--results", default=str(ROOT / "results/results.pkl"))
parser.add_argument("--outdir", default=str(ROOT / "results"))
parser.add_argument("-k", "--panel-size", type=int, default=10)
parser.add_argument("--top", type=int, default=10, help="how many top genes to show per method")
parser.add_argument("--min-freq", type=float, default=0.5, help="'often selected' = in at least this fraction of repeats")
args = parser.parse_args()

df = pd.DataFrame(pickle.load(open(args.results, "rb")))
df = df[df.panel_size == args.panel_size]
n_rep = df.repeat.nunique()
NAMES = {"ttest": "t-test", "l1_logistic": "L1-logistic", "random_forest": "Random forest"}

freq = {}
for m in NAMES:
    c = Counter(g for genes in df[df.method == m].genes for g in genes)
    freq[m] = pd.Series(c) / n_rep

table = pd.DataFrame(freq).fillna(0).rename(columns=NAMES)

print(f"Panel size k = {args.panel_size}, {n_rep} repeats")
for col in NAMES.values():
    top = table[col].sort_values(ascending=False).head(args.top)
    print(f"\nTop {args.top} genes for {col} (fraction of repeats in which the gene was selected):")
    print(top.round(2).to_string())

often = (table >= args.min_freq)
consensus = table[often.sum(axis=1) >= 2].copy()
consensus["n_methods"] = often.sum(axis=1)[consensus.index]
print(f"\nGenes selected in >= {args.min_freq:.0%} of repeats by 2+ methods:")
print(consensus.sort_values(["n_methods"], ascending=False).round(2).to_string() if len(consensus) else "  (none)")
print(f"\nNumber of genes selected often (>= {args.min_freq:.0%}) per method:", often.sum().to_dict())

outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
out = outdir / f"top_genes_k{args.panel_size}.csv"
table.sort_values(list(NAMES.values()), ascending=False).to_csv(out)
print("\nSaved", out)
