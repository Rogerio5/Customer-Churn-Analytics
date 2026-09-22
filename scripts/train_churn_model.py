from pathlib import Path

import joblib
import pandas as pd

from customer_churn.features.builder import ChurnFeatureBuilder
from customer_churn.models.logistic import ChurnLogisticRegressionBuilder

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)

ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "churn_model.joblib"


def main() -> None:
    dataframe = pd.read_csv(DATA_PATH)

    dataset = ChurnFeatureBuilder().build(dataframe)

    model = ChurnLogisticRegressionBuilder().build()

    # A Logistic Regression foi selecionada na versão 1.1
    # pelo maior ROC-AUC médio em validação cruzada
    # estratificada com 5 folds no conjunto de desenvolvimento.
    #
    # Após a seleção, o modelo foi avaliado uma única vez
    # no conjunto de teste final preservado. O artefato
    # operacional é então treinado com toda a base disponível
    # para utilização na previsão de novos clientes.
    model.fit(
        dataset.features,
        dataset.target,
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("=" * 70)
    print("CUSTOMER CHURN - FINAL MODEL TRAINING")
    print("=" * 70)
    print("MODEL=Logistic Regression")
    print(f"TRAIN_ROWS={len(dataset.features)}")
    print(f"MODEL_PATH={MODEL_PATH}")
    print("CHURN_MODEL_TRAINING=PASS")


if __name__ == "__main__":
    main()