from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class LogisticFeatureContribution:
    feature: str
    transformed_value: float
    coefficient: float
    contribution: float


@dataclass(frozen=True)
class LogisticCustomerExplanation:
    intercept: float
    decision_score: float
    contributions: tuple[LogisticFeatureContribution, ...]


class ChurnLogisticExplainer:
    """Explain a Logistic Regression churn prediction using feature contributions."""

    def explain(
        self,
        pipeline: Pipeline,
        features: pd.DataFrame,
    ) -> LogisticCustomerExplanation:
        if len(features) != 1:
            raise ValueError(
                "Logistic customer explanation requires exactly one row."
            )

        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        if not isinstance(model, LogisticRegression):
            raise TypeError(
                "Logistic explainer requires a LogisticRegression model."
            )

        transformed = np.asarray(
            preprocessor.transform(features),
            dtype=float,
        )

        feature_names = tuple(
            preprocessor.get_feature_names_out()
        )

        coefficients = np.asarray(
            model.coef_[0],
            dtype=float,
        )

        row = transformed[0]

        if len(row) != len(coefficients):
            raise ValueError(
                "Transformed features and model coefficients have "
                "different dimensions."
            )

        contribution_values = row * coefficients

        contributions = tuple(
            LogisticFeatureContribution(
                feature=str(feature),
                transformed_value=float(value),
                coefficient=float(coefficient),
                contribution=float(contribution),
            )
            for feature, value, coefficient, contribution in zip(
                feature_names,
                row,
                coefficients,
                contribution_values,
                strict=True,
            )
        )

        intercept = float(model.intercept_[0])

        decision_score = float(
            intercept + contribution_values.sum()
        )

        return LogisticCustomerExplanation(
            intercept=intercept,
            decision_score=decision_score,
            contributions=contributions,
        )