# PICO Classification — Machine Learning for Literature Screening in Meta-Analyses

Automatic recognition of PICO elements in the abstracts of scientific papers, to speed up
the screening phase of systematic reviews and meta-analyses.

---

## 1. Motivation

A meta-analysis starts with a literature search that can return thousands of candidate
papers, of which only a small fraction are actually relevant. Screening them by hand is
the single most time-consuming step of the process, and it has to be repeated for every
new research question.

The PICO framework is the standard way clinical researchers state an inclusion
criterion. If the sentences of an abstract can be automatically tagged with the PICO
element they express, a reviewer can filter a large corpus by *"show me all papers whose
Population is adults with type 2 diabetes"* instead of reading every abstract in full.

This project builds and compares classifiers that perform exactly that tagging, at the
sentence level.

---

## 2. The PICO framework

PICO is a framework used to formulate clinical research questions in a clear and
structured way. It is widely applied in meta-analyses and systematic reviews in
biostatistics and epidemiology.

| Letter | Meaning | Example |
|---|---|---|
| **P** | Population — the patient group of interest | adults with type 2 diabetes |
| **I** | Intervention — the treatment under study | administration of metformin |
| **C** | Comparison — the control or comparator | placebo, or another drug |
| **O** | Outcome — the measured result | reduction of fasting blood glucose |

A **T** is sometimes added (PICOT) for *Time* or *Type of study*, e.g. the duration of the
study or the restriction to randomised controlled trials.

In a meta-analysis, PICO is used to:

1. define the inclusion/exclusion criteria for the studies to be analysed;
2. structure the bibliographic search so that the most relevant studies are identified;
3. guarantee consistency when comparing studies, making results more comparable.

---

## 3. Dataset

The corpus consists of sentences extracted from structured abstracts, each carrying the
section heading it appeared under (e.g. `METHODS`, `MAIN OUTCOME MEASURES`). The raw data
lives in an Excel file with two columns used by the code:

| Column | Type | Description |
|---|---|---|
| `Text` | string | the sentence |
| `Category` | int (0/1) | binary target for the element being modelled |

`data_utils.load_dataset()` is the single place that reads this file and enforces the
schema: it checks both columns are present, drops rows with missing `Text`/`Category`,
and casts `Category` to `int`. Every training script goes through it, so the label is
already a clean 0/1 integer by the time it reaches any model — no `LabelEncoder` needed.

> **Note on data availability.** The full corpus is not distributed with this repository.
> A small synthetic sample with the same schema is provided in `data/sample_data.xlsx` so
> that the pipelines can be run end to end. See `data/README.md`.

### 3.1 Label mapping

The original headings are heterogeneous: many different strings denote the same PICO
element. All of them were read, analysed and grouped as follows.

| Group | Original headings |
|---|---|
| **R** — Results | `RESULTS`, `RESULT(S)`, `FINDINGS`, `MAIN RESULTS` |
| **A** — Aim / Background | `OBJECTIVES`, `BACKGROUND AND PURPOSE`, `OBJECTIVE`, `CONTEXT`, `AIM`, `IMPORTANCE`, `BACKGROUND`, `PURPOSE`, `AIMS`, `BACKGROUND AND OBJECTIVE`, `STUDY OBJECTIVE`, `PURPOSE/OBJECTIVES`, `RATIONALE` |
| **M** — Methods / Measures | `DESIGN AND SETTING`, `METHOD`, `METHODS`, `SETTING`, `DESIGN`, `METHODS AND ANALYSIS`, `STUDY DESIGN`, `MATERIAL AND METHOD`, `RESEARCH DESIGN AND METHODS` |
| **P** — Participants | `PARTICIPANTS`, `SUBJECTS`, `PATIENTS`, `PATIENT(S)`, `POPULATION`, `SAMPLE`, `STUDY SAMPLE`, `WHAT IS ALREADY KNOWN ABOUT THIS SUBJECT`, `TYPES OF PARTICIPANTS` |
| **I** — Intervention | `INTERVENTION`, `INTERVENTION(S)`, `INTERVENTIONS` |
| **C** — Conclusion | `CONCLUSION AND DISCUSSION`, `CONCLUSION`, `CONCLUSIONS AND RELEVANCE`, `CONCLUSION(S)`, `CONCLUSIONS AND CLINICAL RELEVANCE`, `DISCUSSION` |
| **O** — Outcome measures | `MEASUREMENTS`, `MAIN OUTCOME MEASURES`, `SECONDARY OUTCOMES`, `MAIN MEASURES`, `OUTCOME MEASURES`, `TREATMENT AND OUTCOME`, `MAIN OUTCOME MEASURE(S)`, `OUTCOMES`, `OUTCOME` |

### 3.2 Label analysis

**P (Population)** — the only group that needed no adjustment: it identifies the study
population exactly. The one exception is `WHAT IS ALREADY KNOWN ABOUT THIS SUBJECT`, which
is sometimes used as an **A** heading and is usually followed by `AIM |A|`. It was kept
under **P**.

**R (Results)** — does not map onto the **O** of PICO, because the information reported
under R is restated in **C** (which corresponds to the PICO outcome). R highlights the
findings of the study, e.g. *"Patients in the epinephrine group also had higher
glycemia"*. It is the most frequent group in the corpus, and a candidate for exclusion.

**I (Intervention)** — describes how the intervention is delivered, e.g. *"Volumes were
fixed to approximate sweat rates and minimize dehydration"*. It is the least frequent
group, which makes it the hardest target and the one most affected by undersampling.

**O (Outcome measures)** — refers to what was measured and how. It does not map directly
onto a single PICO parameter but is tied to results and measurement reporting.

**A (Aim and Background)** — states the background and the purpose of the study. It
sometimes carries P-related information that is later stated properly under P; this is a
source of confusion for a classifier, but overall A defines the study context.

**M (Methods)** — the most heterogeneous group: it covers the type of analysis, how it was
conducted, and sometimes the study design itself (*"Prospective randomized study"*).

### 3.3 The derived **G** label (Comparison)

The **C** of PICO (Comparison) has no corresponding heading in the source data. However, a
keyword search over the corpus (`placebo`, `versus`, `compared`, `usual care`,
`no treatment`) surfaced a large number of sentences that do express a comparison.

A new label **G** was therefore derived: all sentences from **M** and **R** containing one
of those comparison cues. This is a heuristic, weakly-supervised label — see
§10 Limitations.

### 3.4 From headings to PICO targets

| PICO element | Source label(s) |
|---|---|
| **P** Population | `P` |
| **I** Intervention | `I` |
| **C** Comparison | `G` (derived) |
| **O** Outcome | `C` ∪ `R` |

---

## 4. Task formulation

Each PICO element is modelled as a **separate binary classification problem** in a
one-vs-rest setting:

- **positive class (1)** — sentences belonging to the target element;
- **negative class (0)** — all other sentences.

This is why every script in this repository trains on two classes, and why the BERT model
is instantiated with `num_labels=2` and evaluated with `average="binary"`. To model a
different PICO element, rebuild the target column and re-run; nothing else changes.

The resulting classes are strongly imbalanced (I is rare, R is abundant), so the training
set is **undersampled to the size of the minority class** by `underSample2Min()`.

---

## 5. Setup and usage

```bash
git clone https://github.com/<user>/pico-classification.git
cd pico-classification
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset in `data/` (or use the provided sample), then, from the repo root:

```bash
# Logistic Regression + Random Forest + SVM, with grid search.
# All three models are tuned with the same GridSearchCV scoring (default: f1),
# so the comparison across them is apples-to-apples. Writes
# results/reports/{model}.txt and results/comparison_classical.md.
python src/train_classification.py --data data/dati_Pico.xlsx --seed 42 \
    --test-size 0.30 --scoring f1 --out results/reports

# AutoML pipeline search (long-running). Exports the best pipeline to
# results/tpot_optimal_pipeline.py; test split is fixed at 30%.
python src/train_tpot.py --data data/dati_Pico.xlsx --seed 42

# BERT fine-tuning (GPU strongly recommended). Splits data/dati_Pico.xlsx into
# train/val/test with one seed, keeps the checkpoint with the best validation
# F1, and evaluates it exactly once on the held-out test set. Writes
# results/reports/bert.txt.
python src/train_bert.py --data data/dati_Pico.xlsx --seed 1702 \
    --epochs 4 --batch-size 16 --max-length 180 --lr 2e-5 \
    --test-size 0.30 --val-size 0.15 --out results/reports
```

`train_tpot.py` does not yet expose `--out`/`--test-size` flags the way the other two
scripts do — its test split is hardcoded to 30%, and it only saves the exported pipeline,
not a text report. This is a known gap, see §11 Future work.

---

## 6. Methods

### 6.1 Shallow learning (`train_classification.py`)

Text is vectorised as a bag of words with `CountVectorizer`; hyperparameters of both the
vectoriser and the classifier are tuned jointly with `GridSearchCV` (5-fold), using the
same scoring metric for every model so the three are comparable.

| Model | Searched hyperparameters |
|---|---|
| Logistic Regression | `C ∈ {0.01, 0.1, 1, 10}`; `ngram_range ∈ {(1,1),(1,2),(1,3)}`; English stop words |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`; same vectoriser grid |
| SVM | `C ∈ {0.01 … 100}`; `kernel ∈ {linear, rbf}`; `gamma ∈ {scale, auto}`; same vectoriser grid |

Each model's best estimator (`grid.best_estimator_`, not the untuned default pipeline) is
what gets evaluated on the test set and reported.

### 6.2 AutoML (`train_tpot.py`)

`TPOTClassifier` with the `TPOT sparse` configuration explores pipelines over the same
sparse bag-of-words representation (`ngram_range=(1,3)`, `min_df=2`, English stop words),
optimising F1 over 5-fold CV. The best pipeline is exported to
`results/tpot_optimal_pipeline.py` and serves as a strong non-neural baseline.

### 6.3 BERT (`train_bert.py`)

`bert-base-uncased` fine-tuned for sequence classification: max length 180 tokens (CLI
`--max-length`), AdamW (`lr=2e-5`, `eps=1e-8`), batch size 16, 4 epochs, and gradient
updates via `zero_grad`/`backward`/`step` each batch. Data is split once, with a single
seed, into train / validation / test; after every epoch the model is evaluated on the
validation set, and the checkpoint with the best validation F1 is kept in memory. That
checkpoint — not necessarily the last epoch's weights — is the one evaluated on the test
set, and only once, at the very end.

---

## 7. Evaluation

Reported per model: accuracy, precision, recall, F1 on the positive class.

**Recall is the metric that matters most here.** In literature screening, a false negative
means a relevant paper is silently dropped from the meta-analysis — a sample-selection bias
in the final synthesis. A false positive only costs a reviewer the time to read and discard
one abstract. A model with 0.95 recall and 0.70 precision is more useful in this setting
than the reverse, and the operating threshold should be chosen accordingly.

`train_classification.py` writes its own three-row table to
`results/comparison_classical.md` on every run. The table below is the full, five-model
comparison and is filled in by hand from `results/reports/` once TPOT and BERT have also
been run for the PICO element in question.

---

## 8. Limitations

- **G is a heuristic label.** Comparison sentences are identified by keyword matching, so
  the label carries both false positives (*"compared with baseline"* is not a control arm)
  and false negatives (comparisons stated without any cue word). Results for the **C**
  element should be read as a lower bound, and manual validation on a sample is the
  natural next step.
- **Undersampling discards data.** Balancing to the minority class means most of the
  corpus is unused when the target element is rare (**I** in particular). Class weighting
  or focal loss would be the alternative worth testing.
- **Sentence-level, single-label.** A sentence can legitimately carry more than one PICO
  element; the current formulation forces one decision per element independently.
- **Section headings as ground truth.** Labels come from the author-written headings of
  structured abstracts, not from expert annotation, so heading conventions that vary by
  journal propagate into the labels.
- **Abstracts only.** Full texts are not processed.

---



