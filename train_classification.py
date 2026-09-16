# -*- coding: utf-8 -*-
"""Classical ML baselines for binary PICO-element classification.

Mirrors train_tpot.py and train_bertDef.py: same CLI args, same
data_utils.load_dataset / underSample2Min, same seed used everywhere.
Logistic Regression, Random Forest and SVM all go through the same
run_model() helper, with the same scoring metric, so the comparison
across models in results/comparison.md is actually apples-to-apples.
"""

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

from data_utils import load_dataset, underSample2Min


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dati_Pico.xlsx")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--scoring", default="f1",
                         help="GridSearchCV scoring, same for every model "
                              "so the comparison across models is fair")
    parser.add_argument("--out", default="results/reports")
    return parser.parse_args()


def run_model(name, pipeline, param_grid, x_train, y_train, x_test, y_test,
              scoring, out_dir, seed):
    """Grid-search `pipeline`, evaluate the best estimator, save the report.

    Returns a dict with the metrics used in the final comparison table.
    """
    print(f"\n=== {name} ===")
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid.fit(x_train, y_train)
    print("Migliori parametri trovati:", grid.best_params_)

    y_pred = grid.best_estimator_.predict(x_test)
    report = classification_report(y_test, y_pred, digits=4)
    accuracy = accuracy_score(y_test, y_pred)
    print(report)
    print("Accuracy:", accuracy)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    slug = name.lower().replace(" ", "_")
    Path(out_dir, f"{slug}.txt").write_text(
        f"Best params: {grid.best_params_}\n\n{report}"
    )

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    pos = report_dict.get("1", report_dict.get(1, {}))
    return {
        "name": name,
        "accuracy": accuracy,
        "precision": pos.get("precision", float("nan")),
        "recall": pos.get("recall", float("nan")),
        "f1": pos.get("f1-score", float("nan")),
    }


def main():
    args = parse_args()

    df = load_dataset(args.data)
    under = underSample2Min(df, "Category", random_state=args.seed)

    x_all = under.loc[:, "Text"]
    y_all = under.loc[:, "Category"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_all,
    )
    x_train, x_test = list(x_train), list(x_test)
    print(f"train={len(x_train)}  test={len(x_test)}")

    vectorizer_grid = {
        "vectorizer__stop_words": ["english"],
        "vectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)],
    }

    results = []

    # --- Logistic Regression ---
    lr_pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", LogisticRegression(random_state=args.seed, max_iter=1000)),
    ])
    lr_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
        **vectorizer_grid,
    }
    results.append(run_model(
        "Logistic Regression", lr_pipeline, lr_grid,
        x_train, y_train, x_test, y_test,
        args.scoring, args.out, args.seed,
    ))

    # --- Random Forest ---
    rf_pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", RandomForestClassifier(random_state=args.seed)),
    ])
    rf_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
        "classifier__max_features": ["sqrt", "log2"],
        **vectorizer_grid,
    }
    results.append(run_model(
        "Random Forest", rf_pipeline, rf_grid,
        x_train, y_train, x_test, y_test,
        args.scoring, args.out, args.seed,
    ))

    # --- SVM ---
    svm_pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", SVC(random_state=args.seed)),
    ])
    svm_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__kernel": ["linear", "rbf"],
        "classifier__gamma": ["scale", "auto"],
        **vectorizer_grid,
    }
    results.append(run_model(
        "SVM", svm_pipeline, svm_grid,
        x_train, y_train, x_test, y_test,
        args.scoring, args.out, args.seed,
    ))

    # --- Summary table, same shape as the one BERT/TPOT results feed into ---
    lines = ["| Model | Accuracy | Precision | Recall | F1 |",
             "|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['f1']:.4f} |"
        )
    comparison_path = Path("results", "comparison_classical.md")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()