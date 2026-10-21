# Same Prediction, Different Genes: Method Choice vs. Sampling Variation

When a gene-expression study reports a short list of "key genes", how much of that list reflects the data
and how much reflects the feature-selection method? This project compares two sources of disagreement on
TCGA breast cancer RNA-seq (primary tumor vs. solid normal tissue):

* **Within-method instability**: the same selector run on different patient resamples.
* **Between-method disagreement**: different selectors run on the *same* train/test split.

Three selectors (univariate t-test, L1-regularized logistic regression, random forest importance) are
evaluated at panel sizes k = 5, 10, 25, 50, 100, always with the same downstream logistic regression.
Gene-panel overlap is measured with the Jaccard index and the chance-corrected Kuncheva index.

## Data

Download from UCSC Xena (TCGA Breast Cancer (BRCA) cohort, `HiSeqV2` gene expression, log2(RSEM+1)):
<https://xenabrowser.net/datapages/?cohort=TCGA%20Breast%20Cancer%20(BRCA)>
and place the expression file at `data/raw/TCGA_BRCA.sampleMap_HiSeqV2`.
Data are not stored in this repository.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python src/01_preprocess.py                    # ~1 min   -> data/processed/*.pkl
python src/02_run_experiment.py --repeats 3    # quick test (~1 min)
python src/02_run_experiment.py                # full run, 50 repeats (~15-30 min) -> results/results.pkl
python src/03_analyze_stability.py             # -> results/within_method.csv, between_method.csv
python src/04_make_figures.py                  # -> figs/*.png
```

## Design notes

* Splits are made at the **patient** level (all samples of a patient stay together) and stratified by whether the
  patient has a normal sample. 50 repeats, 80% train / 20% test (`--train-size` changes the fraction).
* Standardization and gene ranking use the training fold only.
* L1 logistic regression keeps only ~30 genes at a fixed penalty, so for each k the smallest `C` on a grid with at
  least k non-zero coefficients is used (no zero-coefficient padding).
* 95% CIs are bootstrap over repeats (2000 resamples), not over individual pairs.

## Limitations

One cohort and one easy task; 80% training splits overlap heavily, so sampling variation is probably
underestimated (smaller `--train-size` is a planned sensitivity check); the three methods optimise different criteria.
