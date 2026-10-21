"""
Step 1 - Preprocess TCGA-BRCA data.

Input : the two files downloaded from UCSC Xena (see README)
          - gene expression matrix  (genes x samples), e.g. TCGA_BRCA.sampleMap_HiSeqV2
          - clinical matrix (not strictly needed; sample type is read from the barcode)
Output: data/processed/X_expr.pkl      samples x genes expression table
        data/processed/y_labels.pkl    1 = primary tumor, 0 = solid tissue normal
        data/processed/patient_ids.pkl patient ID for every sample (needed for patient-level splits)

TCGA barcode example:  TCGA-A7-A0CE-11A-...
                        fields 1-3  -> patient ID   (TCGA-A7-A0CE)
                        field 4     -> sample type  (01 = primary tumor, 11 = solid normal, 06 = metastatic)
"""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--expr", default=str(ROOT / "data/raw/TCGA_BRCA.sampleMap_HiSeqV2"),
                    help="path to the Xena HiSeqV2 expression file (tab separated)")
parser.add_argument("--outdir", default=str(ROOT / "data/processed"))
args = parser.parse_args()
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

print("Loading expression matrix (this takes ~10-30 s)...")
expr = pd.read_csv(args.expr, sep="\t", index_col=0)      # rows = genes, columns = samples
print("Expression matrix shape (genes x samples):", expr.shape)


def barcode_type(barcode: str) -> str:
    code = barcode.split("-")[3][:2]
    return {"01": "Primary Tumor", "11": "Solid Tissue Normal", "06": "Metastatic"}.get(code, "Other")


sample_type = pd.Series({s: barcode_type(s) for s in expr.columns})
print("\nAll sample types in the file:\n", sample_type.value_counts())

# Keep only tumor and normal (drop metastatic etc.)
keep = sample_type[sample_type.isin(["Primary Tumor", "Solid Tissue Normal"])].index
labels = sample_type[keep].map({"Primary Tumor": 1, "Solid Tissue Normal": 0})
patient_ids = pd.Series(["-".join(s.split("-")[:3]) for s in keep], index=keep)

# Transpose: machine learning wants samples as rows, genes as columns
X = expr[keep].T
X.index.name = "sample"

n_both = len(set(patient_ids[labels == 1]) & set(patient_ids[labels == 0]))
print(f"\nKept {len(keep)} samples from {patient_ids.nunique()} patients "
      f"({int(labels.sum())} tumor / {int((labels == 0).sum())} normal)")
print(f"Patients that have BOTH a tumor and a normal sample: {n_both}")

X.to_pickle(outdir / "X_expr.pkl")
labels.to_pickle(outdir / "y_labels.pkl")
patient_ids.to_pickle(outdir / "patient_ids.pkl")
print("Saved to", outdir, "| X shape (samples x genes):", X.shape)
