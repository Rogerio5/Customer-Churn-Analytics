from pathlib import Path

import pandas as pd

from customer_churn.analytics.intelligence import (
    ChurnIntelligenceBuilder,
)
from customer_churn.features.builder import (
    ChurnFeatureBuilder,
)
from customer_churn.features.splitter import (
    ChurnDatasetSplitter,
)
from customer_churn.models.logistic import (
    ChurnLogisticRegressionBuilder,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)

INTELLIGENCE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "churn_customer_intelligence.csv"
)

BI_PATH = (
    ROOT
    / "data"
    / "processed"
    / "churn_customer_intelligence_bi.csv"
)

POWERBI_PATH = (
    ROOT
    / "dashboards"
    / "powerbi"
    / "data"
    / "churn_customer_intelligence_bi.csv"
)


TEST_SIZE = 0.20
HOLDOUT_RANDOM_STATE = 20260920


BI_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "OnlineSecurity",
    "PaperlessBilling",
]


def main() -> None:
    dataframe = pd.read_csv(
        DATA_PATH
    )

    dataset = ChurnFeatureBuilder().build(
        dataframe
    )

    split = ChurnDatasetSplitter(
        test_size=TEST_SIZE,
        random_state=HOLDOUT_RANDOM_STATE,
    ).split(dataset)

    model = (
        ChurnLogisticRegressionBuilder()
        .build()
    )

    # O modelo da fila é treinado somente
    # com o conjunto de desenvolvimento.
    model.fit(
        split.x_train,
        split.y_train,
    )

    probabilities = model.predict_proba(
        split.x_test
    )[:, 1]

    intelligence = (
        ChurnIntelligenceBuilder()
        .build(
            split.customer_ids_test,
            probabilities,
        )
    )

    intelligence.to_csv(
        INTELLIGENCE_PATH,
        index=False,
    )

    source = (
        dataframe
        .set_index("customerID")
    )

    test_customer_ids = (
        split.customer_ids_test
        .astype(str)
        .tolist()
    )

    customer_details = (
        source
        .loc[
            test_customer_ids,
            BI_COLUMNS + ["Churn"],
        ]
        .reset_index()
        .rename(
            columns={
                "customerID": "customer_id",
                "Churn": "actual_churn",
            }
        )
    )

    bi = intelligence.merge(
        customer_details,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    expected_rows = len(
        split.x_test
    )

    if len(bi) != expected_rows:
        raise RuntimeError(
            "Retention BI row count does not match "
            "the final holdout size."
        )

    if bi["customer_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate customer IDs found in retention BI."
        )

    if bi.isna().any().any():
        missing = (
            bi.columns[
                bi.isna().any()
            ]
            .tolist()
        )

        raise RuntimeError(
            "Missing values found after retention BI merge: "
            f"{missing}"
        )

    BI_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    POWERBI_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bi.to_csv(
        BI_PATH,
        index=False,
    )

    bi.to_csv(
        POWERBI_PATH,
        index=False,
    )

    risk_counts = (
        bi["risk_level"]
        .value_counts()
        .reindex(
            [
                "LOW",
                "MEDIUM",
                "HIGH",
            ],
            fill_value=0,
        )
    )

    priority_counts = (
        bi["retention_priority"]
        .value_counts()
        .reindex(
            [
                "MONITOR",
                "ENGAGE",
                "URGENT",
            ],
            fill_value=0,
        )
    )

    print()
    print("=" * 78)
    print("CUSTOMER CHURN - RETENTION QUEUE V1.1")
    print("=" * 78)

    print()
    print("MODEL=Logistic Regression")
    print(
        f"HOLDOUT_RANDOM_STATE="
        f"{HOLDOUT_RANDOM_STATE}"
    )
    print(
        f"DEVELOPMENT_ROWS="
        f"{len(split.x_train)}"
    )
    print(
        f"RETENTION_ROWS="
        f"{len(split.x_test)}"
    )

    print()
    print("RISK DISTRIBUTION")
    print("-" * 78)

    for level in [
        "LOW",
        "MEDIUM",
        "HIGH",
    ]:
        print(
            f"{level}="
            f"{int(risk_counts[level])}"
        )

    print()
    print("RETENTION PRIORITY")
    print("-" * 78)

    for priority in [
        "MONITOR",
        "ENGAGE",
        "URGENT",
    ]:
        print(
            f"{priority}="
            f"{int(priority_counts[priority])}"
        )

    print()
    print(
        f"INTELLIGENCE_PATH="
        f"{INTELLIGENCE_PATH}"
    )
    print(
        f"BI_PATH={BI_PATH}"
    )
    print(
        f"POWERBI_PATH="
        f"{POWERBI_PATH}"
    )

    print()
    print(
        "CHURN_RETENTION_QUEUE_V1_1=PASS"
    )


if __name__ == "__main__":
    main()