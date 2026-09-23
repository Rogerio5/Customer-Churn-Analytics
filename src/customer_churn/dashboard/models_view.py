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
    accuracy = float(final["accuracy"]) * 100
    precision = float(final["precision"]) * 100
    recall = float(final["recall"]) * 100
    f1 = float(final["f1"]) * 100

    tn = int(final["true_negative"])
    fp = int(final["false_positive"])
    fn = int(final["false_negative"])
    tp = int(final["true_positive"])

    left, right = st.columns([1.05, 0.95], gap="large")

    with left, st.container(border=True):
        st.markdown("#### ✅ Modelo selecionado")
        st.markdown(f"### {selected_model}")
        st.write(
            "Selecionado pelo maior ROC-AUC médio na "
            "cross-validation."
        )
        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC CV", f"{selected_cv_auc:.4f}")
        c2.metric("ROC-AUC final", f'{final["roc_auc"]:.4f}')
        st.caption(
            f"Avaliado em {selection['final_test_rows']:,} clientes "
            "que não participaram da seleção."
        )

    with right, st.container(border=True):
        st.markdown("#### Resultado em linguagem simples")
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        c1.metric("Previsões corretas", f"{accuracy:.2f}%")
        c2.metric("Precisão dos alertas", f"{precision:.2f}%")
        c3.metric("Churns encontrados", f"{recall:.2f}%")
        c4.metric("Equilíbrio F1", f"{f1:.2f}%")

    a1, a2 = st.columns(2, gap="large")

    with a1, st.container(border=True):
        st.markdown("#### ✅ Acertos")
        st.write(
            f"**{tn}** clientes sem churn e **{tp}** clientes "
            "com churn foram classificados corretamente."
        )
        st.success(f"Total correto: {tn + tp} clientes.")

    with a2, st.container(border=True):
        st.markdown("#### ⚠️ Erros")
        st.write(
            f"**{fp}** falsos alertas e **{fn}** churns "
            "não identificados no threshold atual."
        )
        st.warning(
            "A previsão auxilia a decisão, mas não representa certeza."
        )

    with st.expander("Como chegamos ao modelo?"):
        st.code(
            f"""
{selection["total_rows"]:,} clientes
        |
        +-- 80% desenvolvimento
        |      {selection["development_rows"]:,}
        |      5-fold cross-validation
        |      Logistic Regression / Random Forest / XGBoost
        |      seleção por ROC-AUC médio
        |
        +-- 20% teste final
               {selection["final_test_rows"]:,}
""",
            language=None,
        )


def render_data_science_view(
    cv,
    final,
    selection,
    shap,
    selected_model,
):
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
            "Desvio": cv["roc_auc_std"].map(
                lambda x: f"{x:.4f}"
            ),
        }
    )

    validation_tab, final_tab, explain_tab, config_tab = st.tabs(
        [
            "🔁 Validação",
            "🎯 Teste final",
            "🧠 Explicabilidade",
            "⚙️ Configuração",
        ]
    )

    with validation_tab:
        left, right = st.columns([1.25, 0.75], gap="large")

        with left:
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                height=185,
            )

        with right:
            chart = (
                cv[["model", "roc_auc_mean"]]
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

    with final_tab:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f'{final["accuracy"]:.4f}')
        c2.metric("Precision", f'{final["precision"]:.4f}')
        c3.metric("Recall", f'{final["recall"]:.4f}')
        c4.metric("F1", f'{final["f1"]:.4f}')
        c5.metric("ROC-AUC", f'{final["roc_auc"]:.4f}')

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
            index=["Real: Não churn", "Real: Churn"],
            columns=[
                "Previsto: Não churn",
                "Previsto: Churn",
            ],
        )

        left, right = st.columns([0.9, 1.1], gap="large")

        with left:
            st.markdown("##### Matriz de confusão")
            st.dataframe(
                confusion,
                width="stretch",
                height=145,
            )

        with right, st.container(border=True):
            st.markdown("##### Threshold atual: 0,50")
            st.write(
                "Abaixo de 50% → não churn. "
                "A partir de 50% → churn."
            )
            st.caption(
                "Reduzir o threshold tende a aumentar Recall, "
                "mas também pode gerar mais falsos alertas."
            )

    with explain_tab:
        st.success(f"Modelo selecionado: {selected_model}")

        if MODEL_PATH.exists():
            model = load_model()
            coefficients = build_coefficients(model)

            positive = (
                coefficients[
                    coefficients["coefficient"] > 0
                ]
                .nlargest(8, "coefficient")
                [["Fator", "coefficient"]]
                .rename(columns={"coefficient": "Coeficiente"})
            )

            negative = (
                coefficients[
                    coefficients["coefficient"] < 0
                ]
                .nsmallest(8, "coefficient")
                [["Fator", "coefficient"]]
                .rename(columns={"coefficient": "Coeficiente"})
            )

            coef_tab, shap_tab = st.tabs(
                ["Logistic Regression", "SHAP complementar"]
            )

            with coef_tab:
                left, right = st.columns(2)
                with left:
                    st.markdown("##### Aumentam o score")
                    st.dataframe(
                        positive,
                        hide_index=True,
                        width="stretch",
                        height=270,
                    )
                with right:
                    st.markdown("##### Reduzem o score")
                    st.dataframe(
                        negative,
                        hide_index=True,
                        width="stretch",
                        height=270,
                    )
                st.caption(
                    "Coeficientes descrevem a direção do score; "
                    "não demonstram causalidade."
                )

            with shap_tab:
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

                left, right = st.columns([0.8, 1.2])
                with left:
                    st.dataframe(
                        shap_display,
                        hide_index=True,
                        width="stretch",
                        height=300,
                    )
                with right:
                    st.bar_chart(
                        shap_display
                        .set_index("Fator")[
                            "Importância SHAP"
                        ]
                    )
                st.caption(
                    "SHAP é uma análise complementar do XGBoost."
                )

    with config_tab:
        left, right = st.columns(2, gap="large")

        with left, st.container(border=True):
            st.markdown("#### Dados e validação")
            st.markdown(
                f"""
- Total: **{selection["total_rows"]:,}**
- Desenvolvimento: **{selection["development_rows"]:,}**
- Teste final: **{selection["final_test_rows"]:,}**
- Holdout: **20%**
- Estratificação: **Churn**
"""
            )

        with right, st.container(border=True):
            st.markdown("#### Cross-validation")
            st.markdown(
                f"""
- Tipo: **StratifiedKFold**
- Folds: **{selection["cv"]["folds"]}**
- Shuffle: **{selection["cv"]["shuffle"]}**
- Random state: `{selection["cv"]["random_state"]}`
- Seleção: **mean ROC-AUC**
- Threshold: **0,50**
"""
            )


def render_models_page():
    cv, final_df, selection, shap = load_data()

    selected_model = selection["selected_model"]
    final = final_df.iloc[0]

    selected_row = cv[
        cv["model"] == selected_model
    ].iloc[0]
    selected_cv_auc = float(
        selected_row["roc_auc_mean"]
    )

    st.markdown(
        '<div class="churn-eyebrow">Validação e interpretação</div>',
        unsafe_allow_html=True,
    )
    st.title("Modelos e Explicabilidade")
    st.caption(
        "Resultados do modelo para negócio e detalhes técnicos "
        "organizados em uma visão compacta."
    )

    model_col, cv_col, final_col, test_col = st.columns(
        [1.35, 1, 1, 1],
        gap="medium",
    )

    with model_col, st.container(border=True):
        st.caption("Modelo selecionado")
        st.markdown(f"### {selected_model}")

    cv_col.metric("ROC-AUC CV", f"{selected_cv_auc:.4f}")
    final_col.metric(
        "ROC-AUC final",
        f'{final["roc_auc"]:.4f}',
    )
    test_col.metric(
        "Teste final",
        f'{int(final["test_rows"]):,}',
    )

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
