import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)

from customer_churn.features.builder import (
    ChurnFeatureBuilder,
)
from customer_churn.features.splitter import (
    ChurnDatasetSplitter,
)
from customer_churn.models.evaluation import (
    ChurnModelEvaluator,
)
from customer_churn.models.logistic import (
    ChurnLogisticRegressionBuilder,
)
from customer_churn.models.random_forest import (
    ChurnRandomForestBuilder,
)
from customer_churn.models.xgboost_model import (
    ChurnXGBoostBuilder,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)

REPORTS_DIR = ROOT / "reports"

CV_REPORT_PATH = (
    REPORTS_DIR
    / "churn_cv_model_comparison.csv"
)

FINAL_REPORT_PATH = (
    REPORTS_DIR
    / "churn_final_test_evaluation.csv"
)

SELECTION_REPORT_PATH = (
    REPORTS_DIR
    / "churn_model_selection_v1_1.json"
)


TEST_SIZE = 0.20

# Novo holdout para a V1.1.
# O split antigo da V1.0 já havia sido utilizado
# para comparar os três modelos.
HOLDOUT_RANDOM_STATE = 20260920

CV_FOLDS = 5
CV_RANDOM_STATE = 42


def build_models():
    return {
        "Logistic Regression": (
            ChurnLogisticRegressionBuilder().build()
        ),
        "Random Forest": (
            ChurnRandomForestBuilder().build()
        ),
        "XGBoost": (
            ChurnXGBoostBuilder().build()
        ),
    }


def main() -> None:
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(
        DATA_PATH
    )

    dataset = ChurnFeatureBuilder().build(
        dataframe
    )

    # --------------------------------------------------------
    # 1. Separar desenvolvimento e teste final.
    # --------------------------------------------------------

    split = ChurnDatasetSplitter(
        test_size=TEST_SIZE,
        random_state=HOLDOUT_RANDOM_STATE,
    ).split(dataset)

    development_rows = len(
        split.x_train
    )

    final_test_rows = len(
        split.x_test
    )

    print()
    print("=" * 78)
    print("CUSTOMER CHURN - MODEL SELECTION V1.1")
    print("=" * 78)

    print()
    print("DATASET")
    print("-" * 78)
    print(
        f"TOTAL_ROWS={len(dataframe)}"
    )
    print(
        f"DEVELOPMENT_ROWS={development_rows}"
    )
    print(
        f"FINAL_TEST_ROWS={final_test_rows}"
    )
    print(
        f"HOLDOUT_RANDOM_STATE="
        f"{HOLDOUT_RANDOM_STATE}"
    )

    # --------------------------------------------------------
    # 2. Cross-validation somente no desenvolvimento.
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    models = build_models()

    cv_records = []

    print()
    print("5-FOLD STRATIFIED CROSS-VALIDATION")
    print("-" * 78)

    for model_name, model in models.items():
        results = cross_validate(
            model,
            split.x_train,
            split.y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,
        )

        record = {
            "model": model_name,
            "cv_folds": CV_FOLDS,
            "accuracy_mean": float(
                results[
                    "test_accuracy"
                ].mean()
            ),
            "accuracy_std": float(
                results[
                    "test_accuracy"
                ].std()
            ),
            "precision_mean": float(
                results[
                    "test_precision"
                ].mean()
            ),
            "precision_std": float(
                results[
                    "test_precision"
                ].std()
            ),
            "recall_mean": float(
                results[
                    "test_recall"
                ].mean()
            ),
            "recall_std": float(
                results[
                    "test_recall"
                ].std()
            ),
            "f1_mean": float(
                results[
                    "test_f1"
                ].mean()
            ),
            "f1_std": float(
                results[
                    "test_f1"
                ].std()
            ),
            "roc_auc_mean": float(
                results[
                    "test_roc_auc"
                ].mean()
            ),
            "roc_auc_std": float(
                results[
                    "test_roc_auc"
                ].std()
            ),
        }

        cv_records.append(
            record
        )

        print()
        print(model_name)
        print(
            "  Accuracy : "
            f"{record['accuracy_mean']:.4f} "
            f"+/- {record['accuracy_std']:.4f}"
        )
        print(
            "  Precision: "
            f"{record['precision_mean']:.4f} "
            f"+/- {record['precision_std']:.4f}"
        )
        print(
            "  Recall   : "
            f"{record['recall_mean']:.4f} "
            f"+/- {record['recall_std']:.4f}"
        )
        print(
            "  F1       : "
            f"{record['f1_mean']:.4f} "
            f"+/- {record['f1_std']:.4f}"
        )
        print(
            "  ROC-AUC  : "
            f"{record['roc_auc_mean']:.4f} "
            f"+/- {record['roc_auc_std']:.4f}"
        )

    cv_dataframe = pd.DataFrame(
        cv_records
    ).sort_values(
        "roc_auc_mean",
        ascending=False,
    )

    cv_dataframe.to_csv(
        CV_REPORT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 3. Selecionar usando APENAS a cross-validation.
    # --------------------------------------------------------

    selected_name = str(
        cv_dataframe.iloc[0][
            "model"
        ]
    )

    selected_cv_auc = float(
        cv_dataframe.iloc[0][
            "roc_auc_mean"
        ]
    )

    print()
    print("=" * 78)
    print("MODEL SELECTION")
    print("=" * 78)
    print(
        f"SELECTED_MODEL={selected_name}"
    )
    print(
        "SELECTION_METRIC="
        "MEAN_CV_ROC_AUC"
    )
    print(
        f"SELECTED_CV_ROC_AUC="
        f"{selected_cv_auc:.4f}"
    )

    # --------------------------------------------------------
    # 4. Só agora tocar no teste final.
    # --------------------------------------------------------

    selected_model = build_models()[
        selected_name
    ]

    selected_model.fit(
        split.x_train,
        split.y_train,
    )

    predictions = selected_model.predict(
        split.x_test
    )

    probabilities = (
        selected_model.predict_proba(
            split.x_test
        )[:, 1]
    )

    evaluator = ChurnModelEvaluator()

    final_metrics = evaluator.evaluate(
        split.y_test,
        predictions,
        probabilities,
    )

    final_record = {
        "model": selected_name,
        "test_rows": final_test_rows,
        "accuracy": final_metrics.accuracy,
        "precision": final_metrics.precision,
        "recall": final_metrics.recall,
        "f1": final_metrics.f1,
        "roc_auc": final_metrics.roc_auc,
        "true_negative": (
            final_metrics.true_negative
        ),
        "false_positive": (
            final_metrics.false_positive
        ),
        "false_negative": (
            final_metrics.false_negative
        ),
        "true_positive": (
            final_metrics.true_positive
        ),
    }

    pd.DataFrame(
        [final_record]
    ).to_csv(
        FINAL_REPORT_PATH,
        index=False,
    )

    selection_report = {
        "version": "1.1",
        "total_rows": len(
            dataframe
        ),
        "development_rows": (
            development_rows
        ),
        "final_test_rows": (
            final_test_rows
        ),
        "test_size": TEST_SIZE,
        "holdout_random_state": (
            HOLDOUT_RANDOM_STATE
        ),
        "cv": {
            "type": (
                "StratifiedKFold"
            ),
            "folds": CV_FOLDS,
            "shuffle": True,
            "random_state": (
                CV_RANDOM_STATE
            ),
        },
        "selection_metric": (
            "mean_cv_roc_auc"
        ),
        "selected_model": (
            selected_name
        ),
        "selected_cv_roc_auc": (
            selected_cv_auc
        ),
        "final_test_metrics": (
            final_record
        ),
    }

    with SELECTION_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selection_report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 78)
    print("FINAL HOLDOUT EVALUATION")
    print("=" * 78)
    print(
        f"MODEL={selected_name}"
    )
    print(
        f"TEST_ROWS={final_test_rows}"
    )
    print(
        f"Accuracy={final_metrics.accuracy:.4f}"
    )
    print(
        f"Precision={final_metrics.precision:.4f}"
    )
    print(
        f"Recall={final_metrics.recall:.4f}"
    )
    print(
        f"F1={final_metrics.f1:.4f}"
    )
    print(
        f"ROC-AUC={final_metrics.roc_auc:.4f}"
    )

    print()
    print(
        "Confusion:",
        f"TN={final_metrics.true_negative}",
        f"FP={final_metrics.false_positive}",
        f"FN={final_metrics.false_negative}",
        f"TP={final_metrics.true_positive}",
    )

    print()
    print(
        f"CV_REPORT={CV_REPORT_PATH}"
    )
    print(
        f"FINAL_REPORT={FINAL_REPORT_PATH}"
    )
    print(
        f"SELECTION_REPORT="
        f"{SELECTION_REPORT_PATH}"
    )

    print()
    print(
        "CHURN_MODEL_SELECTION_V1_1=PASS"
    )


if __name__ == "__main__":
    main()