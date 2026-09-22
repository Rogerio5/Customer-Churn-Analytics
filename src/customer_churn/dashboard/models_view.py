import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from customer_churn.dashboard.predict_view import humanize_feature

ROOT = Path(__file__).resolve().parents[3]

CV_PATH = ROOT / "reports/churn_cv_model_comparison.csv"
FINAL_PATH = ROOT / "reports/churn_final_test_evaluation.csv"
SELECTION_PATH = ROOT / "reports/churn_model_selection_v1_1.json"
SHAP_PATH = ROOT / "dashboards/powerbi/data/churn_shap_importance.csv"
MODEL_PATH = ROOT / "artifacts/churn_model.joblib"


@st.cache_data
def load_data():
    cv = pd.read_csv(CV_PATH)
    final = pd.read_csv(FINAL_PATH)

    with SELECTION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        selection = json.load(file)

    shap = pd.read_csv(SHAP_PATH)

    return cv, final, selection, shap


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_coefficients(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]

    result = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "coefficient": estimator.coef_[0],
        }
    )

    result["Fator"] = result["feature"].map(humanize_feature)
    result["Magnitude"] = result["coefficient"].abs()

    return result


def render_business_view(
    final,
    selection,
    selected_model,
    selected_cv_auc,
):
    st.subheader("Entenda o resultado")

    st.caption(
        "Esta parte foi feita para quem não trabalha "
        "diretamente com Machine Learning."
    )

    with st.container(border=True):
        st.markdown(
            f"""
### ✅ Modelo escolhido: {selected_model}

Foram comparados **Logistic Regression, Random Forest e XGBoost**.

O modelo escolhido apresentou o maior resultado médio na
cross-validation:

**ROC-AUC médio: {selected_cv_auc:.4f}**

Depois disso, ele foi testado em
**{selection["final_test_rows"]:,} clientes que não participaram
da escolha do modelo**.

**ROC-AUC no teste final: {final["roc_auc"]:.4f}**
"""
        )

        st.caption(
            "Em termos simples, o ROC-AUC mede o quanto o modelo "
            "consegue ordenar clientes de maior e menor risco."
        )

    st.markdown("### Resultado em linguagem simples")

    accuracy = float(final["accuracy"]) * 100
    precision = float(final["precision"]) * 100
    recall = float(final["recall"]) * 100
    f1 = float(final["f1"]) * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Previsões corretas",
        f"{accuracy:.2f}%",
    )
    c1.caption(
        "Percentual total de classificações corretas."
    )

    c2.metric(
        "Precisão dos alertas",
        f"{precision:.2f}%",
    )
    c2.caption(
        "Entre os alertas de churn, quantos realmente tiveram churn."
    )

    c3.metric(
        "Churns encontrados",
        f"{recall:.2f}%",
    )
    c3.caption(
        "Entre os clientes que tiveram churn, quantos foram encontrados."
    )

    c4.metric(
        "Equilíbrio",
        f"{f1:.2f}%",
    )
    c4.caption(
        "Equilíbrio entre Precision e Recall."
    )

    tn = int(final["true_negative"])
    fp = int(final["false_positive"])
    fn = int(final["false_negative"])
    tp = int(final["true_positive"])

    st.markdown("### Traduzindo em clientes")

    left, right = st.columns(2)

    with left, st.container(border=True):
        st.markdown("#### ✅ O modelo acertou")

        st.write(
            f"**{tn} clientes** sem churn foram "
            "identificados corretamente."
        )

        st.write(
            f"**{tp} clientes** com churn foram "
            "identificados corretamente."
        )

        st.success(
            f"Total correto: {tn + tp} clientes."
        )

    with right, st.container(border=True):
        st.markdown("#### ⚠️ O modelo errou")

        st.write(
            f"**{fp} clientes** receberam alerta de churn, "
            "mas não tiveram churn."
        )

        st.write(
            f"**{fn} clientes** tiveram churn, mas não foram "
            "identificados pelo limite atual."
        )

        st.warning(
            "A previsão auxilia a decisão, mas não representa certeza."
        )

    st.markdown("### Como chegamos ao modelo?")

    st.code(
        f"""
{selection["total_rows"]:,} clientes
        |
        +-- 80% desenvolvimento
        |      {selection["development_rows"]:,} clientes
        |
        |      5-fold cross-validation
        |        +-- Logistic Regression
        |        +-- Random Forest
        |        +-- XGBoost
        |
        |      escolha pelo ROC-AUC médio
        |
        +-- 20% teste final
               {selection["final_test_rows"]:,} clientes
""",
        language=None,
    )

    st.caption(
        "O conjunto de teste final só foi utilizado "
        "depois da seleção do modelo."
    )


def render_data_science_view(
    cv,
    final,
    selection,
    shap,
    selected_model,
):
    st.subheader("Ciência de Dados")

    st.caption(
        "Aqui ficam os detalhes técnicos da seleção, "
        "avaliação e explicabilidade."
    )

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    st.markdown("### 1. Cross-Validation")

    cv = cv.sort_values(
        "roc_auc_mean",
        ascending=False,
    ).copy()

    table = pd.DataFrame(
        {
            "Modelo": cv["model"],
            "Accuracy": cv["accuracy_mean"].map(
                lambda x: f"{x:.4f}"
            ),
            "Precision": cv["precision_mean"].map(
                lambda x: f"{x:.4f}"
            ),
            "Recall": cv["recall_mean"].map(
                lambda x: f"{x:.4f}"
            ),
            "F1": cv["f1_mean"].map(
                lambda x: f"{x:.4f}"
            ),
            "ROC-AUC médio": cv["roc_auc_mean"].map(
                lambda x: f"{x:.4f}"
            ),
            "Desvio ROC-AUC": cv["roc_auc_std"].map(
                lambda x: f"{x:.4f}"
            ),
        }
    )

    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
    )

    chart = (
        cv[
            [
                "model",
                "roc_auc_mean",
            ]
        ]
        .set_index("model")
        .rename(
            columns={
                "roc_auc_mean": "ROC-AUC médio",
            }
        )
    )

    st.bar_chart(chart)

    logistic_auc = float(
        cv.loc[
            cv["model"] == "Logistic Regression",
            "roc_auc_mean",
        ].iloc[0]
    )

    xgb_auc = float(
        cv.loc[
            cv["model"] == "XGBoost",
            "roc_auc_mean",
        ].iloc[0]
    )

    st.info(
        "Logistic Regression e XGBoost ficaram próximos: "
        f"diferença de ROC-AUC médio = "
        f"{logistic_auc - xgb_auc:.4f}."
    )

    st.caption(
        "O desvio mostra a variação entre os cinco folds. "
        "Não é intervalo de confiança."
    )

    # --------------------------------------------------------
    # Test final
    # --------------------------------------------------------

    st.markdown("### 2. Teste final")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f'{final["accuracy"]:.4f}',
    )
    c2.metric(
        "Precision",
        f'{final["precision"]:.4f}',
    )
    c3.metric(
        "Recall",
        f'{final["recall"]:.4f}',
    )
    c4.metric(
        "F1",
        f'{final["f1"]:.4f}',
    )
    c5.metric(
        "ROC-AUC",
        f'{final["roc_auc"]:.4f}',
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    st.markdown("### 3. Matriz de Confusão")

    confusion = pd.DataFrame(
        [
            [
                int(final["true_negative"]),
                int(final["false_positive"]),
            ],
            [
                int(final["false_negative"]),
                int(final["true_positive"]),
            ],
        ],
        index=[
            "Real: Não churn",
            "Real: Churn",
        ],
        columns=[
            "Previsto: Não churn",
            "Previsto: Churn",
        ],
    )

    st.dataframe(
        confusion,
        width="stretch",
    )

    with st.expander("Entender TN, FP, FN e TP"):
        st.markdown(
            f"""
- **TN = {int(final["true_negative"])}**:
  não churn corretamente identificado.
- **FP = {int(final["false_positive"])}**:
  falso alerta.
- **FN = {int(final["false_negative"])}**:
  churn não identificado.
- **TP = {int(final["true_positive"])}**:
  churn corretamente identificado.
"""
        )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    st.markdown("### 4. Threshold")

    with st.container(border=True):
        st.markdown(
            """
O threshold atual de classificação é **0,50**.

- abaixo de 50% → classe não churn;
- a partir de 50% → classe churn.

Alterar esse limite muda **Precision, Recall, F1 e a matriz
de confusão**.

Em retenção, um threshold menor tende a encontrar mais clientes
em risco, aumentando Recall, porém também pode aumentar falsos
alertas.
"""
        )

    st.caption(
        "ROC-AUC avalia o ranking de risco em vários thresholds, "
        "e por isso não depende apenas do limite de 0,50."
    )

    # --------------------------------------------------------
    # Logistic coefficients
    # --------------------------------------------------------

    st.markdown(
        "### 5. Explicabilidade da Logistic Regression"
    )

    st.success(
        f"Modelo selecionado: {selected_model}"
    )

    if MODEL_PATH.exists():
        model = load_model()

        coefficients = build_coefficients(
            model
        )

        positive = (
            coefficients[
                coefficients["coefficient"] > 0
            ]
            .nlargest(
                8,
                "coefficient",
            )
            [
                [
                    "Fator",
                    "coefficient",
                ]
            ]
            .rename(
                columns={
                    "coefficient": "Coeficiente",
                }
            )
        )

        negative = (
            coefficients[
                coefficients["coefficient"] < 0
            ]
            .nsmallest(
                8,
                "coefficient",
            )
            [
                [
                    "Fator",
                    "coefficient",
                ]
            ]
            .rename(
                columns={
                    "coefficient": "Coeficiente",
                }
            )
        )

        positive["Coeficiente"] = (
            positive["Coeficiente"].round(4)
        )

        negative["Coeficiente"] = (
            negative["Coeficiente"].round(4)
        )

        left, right = st.columns(2)

        with left:
            st.markdown(
                "#### ⬆ Associados a maior score"
            )
            st.dataframe(
                positive,
                hide_index=True,
                width="stretch",
            )

        with right:
            st.markdown(
                "#### ⬇ Associados a menor score"
            )
            st.dataframe(
                negative,
                hide_index=True,
                width="stretch",
            )

        st.caption(
            "Coeficientes positivos aumentam o score do modelo "
            "e negativos reduzem. Isso não representa causalidade."
        )

    # --------------------------------------------------------
    # SHAP complementary
    # --------------------------------------------------------

    st.markdown(
        "### 6. Análise complementar — XGBoost + SHAP"
    )

    st.info(
        "O XGBoost não foi o modelo selecionado na V1.1. "
        "O SHAP é mantido como análise complementar "
        "de explicabilidade em modelos baseados em árvores."
    )

    shap_display = (
        shap.rename(
            columns={
                "feature": "Fator",
                "mean_abs_shap": "Importância SHAP",
            }
        )
        .sort_values(
            "Importância SHAP",
            ascending=False,
        )
    )

    shap_display["Importância SHAP"] = (
        shap_display["Importância SHAP"]
        .astype(float)
        .round(4)
    )

    left, right = st.columns(
        [2, 3]
    )

    with left:
        st.dataframe(
            shap_display,
            hide_index=True,
            width="stretch",
        )

    with right:
        st.bar_chart(
            shap_display
            .set_index("Fator")[
                "Importância SHAP"
            ]
        )

    st.caption(
        "Mean absolute SHAP mede importância preditiva "
        "no XGBoost. Não demonstra causalidade."
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    st.markdown(
        "### 7. Configuração reproduzível"
    )

    with st.expander(
        "Ver detalhes técnicos"
    ):
        st.markdown(
            f"""
**Dataset**

- Total: {selection["total_rows"]:,}
- Desenvolvimento: {selection["development_rows"]:,}
- Teste final: {selection["final_test_rows"]:,}

**Holdout**

- Test size: 20%
- Estratificação: churn
- Random state: `{selection["holdout_random_state"]}`

**Cross-validation**

- StratifiedKFold
- Folds: `{selection["cv"]["folds"]}`
- Shuffle: `{selection["cv"]["shuffle"]}`
- Random state: `{selection["cv"]["random_state"]}`

**Seleção**

- Critério: mean cross-validation ROC-AUC
- Modelo: {selected_model}

**Classificação**

- Threshold: `0.50`
"""
        )


def render_models_page():
    cv, final_df, selection, shap = load_data()

    selected_model = selection[
        "selected_model"
    ]

    final = final_df.iloc[0]

    selected_row = cv[
        cv["model"] == selected_model
    ].iloc[0]

    selected_cv_auc = float(
        selected_row["roc_auc_mean"]
    )

    st.title(
        "Modelos e Explicabilidade"
    )

    st.caption(
        "Uma visão simples para negócio e uma visão "
        "técnica para Ciência de Dados."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Modelo selecionado",
        selected_model,
    )

    c2.metric(
        "ROC-AUC CV",
        f"{selected_cv_auc:.4f}",
    )

    c3.metric(
        "ROC-AUC final",
        f'{final["roc_auc"]:.4f}',
    )

    c4.metric(
        "Teste final",
        f'{int(final["test_rows"]):,}',
    )

    st.divider()

    business_tab, data_science_tab = st.tabs(
        [
            "👤 Entender o resultado",
            "🧪 Ciência de Dados",
        ]
    )

    with business_tab:
        render_business_view(
            final=final,
            selection=selection,
            selected_model=selected_model,
            selected_cv_auc=selected_cv_auc,
        )

    with data_science_tab:
        render_data_science_view(
            cv=cv,
            final=final,
            selection=selection,
            shap=shap,
            selected_model=selected_model,
        )


if __name__ == "__main__":
    render_models_page()