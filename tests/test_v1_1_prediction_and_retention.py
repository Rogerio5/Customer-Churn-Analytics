from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from customer_churn.dashboard.predict_view import (
    contribution_dataframe,
    humanize_feature,
    operational_guidance,
    readable_factor_list,
    top_factor_labels,
)
from customer_churn.features.builder import ChurnFeatureBuilder
from customer_churn.models.logistic_explainability import (
    ChurnLogisticExplainer,
    LogisticFeatureContribution,
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "artifacts" / "churn_model.joblib"

CLEAN_DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)

RETENTION_PATH = (
    ROOT
    / "data"
    / "processed"
    / "churn_customer_intelligence.csv"
)


@pytest.fixture
def sample_contributions():
    return (
        LogisticFeatureContribution(
            feature="Contract_Month-to-month",
            transformed_value=1.0,
            coefficient=1.20,
            contribution=1.20,
        ),
        LogisticFeatureContribution(
            feature="TechSupport_No",
            transformed_value=1.0,
            coefficient=0.80,
            contribution=0.80,
        ),
        LogisticFeatureContribution(
            feature="Contract_Two year",
            transformed_value=1.0,
            coefficient=-1.10,
            contribution=-1.10,
        ),
        LogisticFeatureContribution(
            feature="tenure",
            transformed_value=0.50,
            coefficient=-0.60,
            contribution=-0.30,
        ),
    )


@pytest.fixture
def logistic_pipeline():
    features = pd.DataFrame(
        {
            "tenure": [
                1,
                3,
                12,
                24,
                48,
                72,
            ],
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "One year",
                "One year",
                "Two year",
                "Two year",
            ],
        }
    )

    target = pd.Series(
        [1, 1, 1, 0, 0, 0],
        name="Churn",
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                ["tenure"],
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["Contract"],
            ),
        ],
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
        ]
    )

    pipeline.fit(
        features,
        target,
    )

    return pipeline, features


def test_humanize_categorical_feature():
    result = humanize_feature(
        "Contract_Month-to-month"
    )

    assert result == (
        "Tipo de contrato = Month-to-month"
    )


def test_humanize_numeric_feature():
    assert humanize_feature(
        "MonthlyCharges"
    ) == "Cobrança mensal"


@pytest.mark.parametrize(
    ("factors", "expected"),
    [
        (
            [],
            "nenhum fator de maior contribuição",
        ),
        (
            ["Contrato"],
            "Contrato",
        ),
        (
            ["Contrato", "Suporte"],
            "Contrato e Suporte",
        ),
        (
            [
                "Contrato",
                "Suporte",
                "Internet",
            ],
            "Contrato, Suporte e Internet",
        ),
    ],
)
def test_readable_factor_list(
    factors,
    expected,
):
    assert readable_factor_list(
        factors
    ) == expected


def test_top_factor_labels_positive(
    sample_contributions,
):
    result = top_factor_labels(
        sample_contributions,
        direction="positive",
        top_n=2,
    )

    assert result == [
        "Tipo de contrato = Month-to-month",
        "Suporte técnico = No",
    ]


def test_top_factor_labels_negative(
    sample_contributions,
):
    result = top_factor_labels(
        sample_contributions,
        direction="negative",
        top_n=2,
    )

    assert result == [
        "Tipo de contrato = Two year",
        "Tempo como cliente",
    ]


def test_contribution_dataframe_positive(
    sample_contributions,
):
    result = contribution_dataframe(
        sample_contributions,
        direction="positive",
        top_n=2,
    )

    assert len(result) == 2

    assert list(result.columns) == [
        "Fator identificado",
        "Efeito na previsão",
    ]

    assert result.iloc[0][
        "Fator identificado"
    ] == (
        "Tipo de contrato = Month-to-month"
    )

    assert result.iloc[0][
        "Efeito na previsão"
    ] == "⬆ Aumentou o risco estimado"


@pytest.mark.parametrize(
    (
        "risk_level",
        "priority",
        "expected_text",
    ),
    [
        (
            "HIGH",
            "URGENT",
            "revisar o cliente",
        ),
        (
            "MEDIUM",
            "ENGAGE",
            "avaliar ações de engajamento",
        ),
        (
            "LOW",
            "MONITOR",
            "monitoramento periódico",
        ),
    ],
)
def test_operational_guidance(
    risk_level,
    priority,
    expected_text,
):
    result = operational_guidance(
        risk_level,
        priority,
    )

    assert f"Prioridade {priority}" in result
    assert expected_text in result


def test_logistic_explanation_rebuilds_decision_score(
    logistic_pipeline,
):
    pipeline, features = logistic_pipeline

    customer = features.iloc[[0]]

    explanation = (
        ChurnLogisticExplainer().explain(
            pipeline,
            customer,
        )
    )

    model_score = float(
        pipeline.decision_function(
            customer
        )[0]
    )

    assert np.isclose(
        explanation.decision_score,
        model_score,
    )

    reconstructed = (
        explanation.intercept
        + sum(
            item.contribution
            for item
            in explanation.contributions
        )
    )

    assert np.isclose(
        reconstructed,
        model_score,
    )


def test_logistic_explanation_requires_one_customer(
    logistic_pipeline,
):
    pipeline, features = logistic_pipeline

    with pytest.raises(
        ValueError,
        match="exactly one row",
    ):
        ChurnLogisticExplainer().explain(
            pipeline,
            features.iloc[:2],
        )


def test_persisted_model_predicts_probability():
    assert MODEL_PATH.exists()

    model = joblib.load(
        MODEL_PATH
    )

    dataframe = pd.read_csv(
        CLEAN_DATA_PATH
    )

    dataset = (
        ChurnFeatureBuilder().build(
            dataframe
        )
    )

    customer = (
        dataset.features.iloc[[0]]
    )

    probability = float(
        model.predict_proba(
            customer
        )[0, 1]
    )

    assert 0.0 <= probability <= 1.0

    assert isinstance(
        model.named_steps["model"],
        LogisticRegression,
    )


def test_retention_dataset_distribution():
    data = pd.read_csv(
        RETENTION_PATH
    )

    assert len(data) == 1409

    counts = (
        data["risk_level"]
        .value_counts()
        .to_dict()
    )

    assert counts == {
        "LOW": 876,
        "MEDIUM": 339,
        "HIGH": 194,
    }


def test_retention_risk_thresholds():
    data = pd.read_csv(
        RETENTION_PATH
    )

    low = data[
        data["risk_level"] == "LOW"
    ]

    medium = data[
        data["risk_level"] == "MEDIUM"
    ]

    high = data[
        data["risk_level"] == "HIGH"
    ]

    assert (
        low["churn_probability"] < 0.30
    ).all()

    assert (
        medium["churn_probability"]
        >= 0.30
    ).all()

    assert (
        medium["churn_probability"]
        < 0.60
    ).all()

    assert (
        high["churn_probability"]
        >= 0.60
    ).all()


def test_retention_priority_mapping():
    data = pd.read_csv(
        RETENTION_PATH
    )

    mapping = {
        "LOW": "MONITOR",
        "MEDIUM": "ENGAGE",
        "HIGH": "URGENT",
    }

    for risk, priority in mapping.items():
        rows = data[
            data["risk_level"] == risk
        ]

        assert (
            rows["retention_priority"]
            == priority
        ).all()