# Same Prediction, Different Genes: Method Choice vs. Sampling Variation

When a gene-expression study reports a short list of "key genes", how much of that list reflects the data
and how much reflects the feature-selection method? This project compares two sources of disagreement on
TCGA breast cancer RNA-seq (primary tumor vs. solid normal tissue):

* **Within-method instability**: the same selector run on different patient resamples.
* **Between-method disagreement**: different selectors run on the *same* train/test split.

Three selectors (univariate t-test, L1-regularized logistic regression, random forest importance) are
evaluated at panel sizes k = 5, 10, 25, 50, 100, always with the same downstream logistic regression.
Gene-panel overlap is measured with the Jaccard index and the chance-corrected Kuncheva consistency index.

## Key results

TCGA-BRCA, 1,211 samples (1,097 tumor, 114 normal) from 1,097 patients, 20,530 genes; 50 patient-level
80/20 splits.

* **Accuracy does not separate the methods.** Mean test AUROC is at least 0.999 for every method and panel
  size, even with 5 genes (worst single split: 0.994).
* **Same method, different patients:** mean Jaccard overlap 0.39-0.88.
* **Different methods, same split:** mean Jaccard overlap 0.00-0.23, with 95% CIs that do not overlap the
  within-method ones.

Mean Jaccard overlap of gene panels (80% training):

| Within method (resamples) | k=5 | k=10 | k=25 | k=50 | k=100 |
|---|---|---|---|---|---|
| t-test | 0.569 | 0.723 | 0.787 | 0.877 | 0.744 |
| L1 logistic | 0.732 | 0.642 | 0.574 | 0.464 | 0.416 |
| Random forest | 0.390 | 0.395 | 0.469 | 0.444 | 0.451 |

| Between methods (same split) | Range over k |
|---|---|
| t-test vs L1 | 0.007-0.054 |
| t-test vs random forest | 0.088-0.226 |
| L1 vs random forest | 0.002-0.080 |

**Sensitivity check (50% training):** within-method overlap drops to 0.25-0.69, still above between-method
overlap (0.02-0.20), so the ordering holds although the gap narrows.

## Data

Download from UCSC Xena (TCGA Breast Cancer (BRCA) cohort, `HiSeqV2` gene expression, log2(RSEM+1)):
<https://xenabrowser.net/datapages/?cohort=TCGA%20Breast%20Cancer%20(BRCA)>
and place the expression file at `data/raw/TCGA.BRCA.sampleMap_HiSeqV2`.
Data are not stored in this repository.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python src/01_preprocess.py --expr data/raw/TCGA.BRCA.sampleMap_HiSeqV2   # ~1 min -> data/processed/*.pkl
python src/02_run_experiment.py --repeats 3    # quick test (~1 min)
python src/02_run_experiment.py                # full run, 50 repeats (~15-30 min) -> results/results.pkl
python src/03_analyze_stability.py             # -> results/within_method.csv, between_method.csv
python src/04_make_figures.py                  # -> figs/*.png
python src/05_top_genes.py                     # -> results/top_genes_k10.csv (most frequently chosen genes)
```

Sensitivity check with 50% training:

```bash
python src/02_run_experiment.py --train-size 0.5 --out results/results_train50.pkl
python src/03_analyze_stability.py --results results/results_train50.pkl --outdir results/train50
```

The within- and between-method tables for this check are in `results/train50/`.

## Design notes

* Splits are made at the **patient** level (all samples of a patient stay together) and stratified by whether the
  patient has a normal sample. 50 repeats, 80% train / 20% test (`--train-size` changes the fraction).
* Standardization and gene ranking use the training fold only.
* L1 logistic regression keeps only ~30 genes at a fixed penalty, so for each k the smallest `C` on a grid with at
  least k non-zero coefficients is used (no zero-coefficient padding). Because `C` differs per k, the L1 panels
  across k are not one nested model.
* 95% CIs are bootstrap over repeats (2000 resamples), not over individual pairs.

## Limitations

One cohort and one easy task (tumor vs. normal). With 80% training, splits overlap heavily, so sampling variation
is probably underestimated; a 50% training check narrows the gap between within- and between-method overlap but does
not reverse it. The three methods optimise different criteria, and the comparison of the two sources of variation
is a contrast of perturbations, not a formal variance decomposition. Next: other cancer types, more methods,
independent cohorts.
