import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)

BI_PATH = (
    ROOT
    / "data"
    / "processed"
    / "churn_customer_intelligence_bi.csv"
)

FINAL_PATH = (
    ROOT
    / "reports"
    / "churn_final_test_evaluation.csv"
)

SELECTION_PATH = (
    ROOT
    / "reports"
    / "churn_model_selection_v1_1.json"
)


@st.cache_data
def load_overview_data():
    data = pd.read_csv(DATA_PATH)
    bi = pd.read_csv(BI_PATH)
    final = pd.read_csv(FINAL_PATH)

    with SELECTION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        selection = json.load(file)

    return data, bi, final, selection


def build_risk_summary(
    bi: pd.DataFrame,
) -> pd.DataFrame:
    order = [
        "HIGH",
        "MEDIUM",
        "LOW",
    ]

    summary = (
        bi.groupby(
            [
                "risk_level",
                "retention_priority",
            ],
            as_index=False,
        )
        .agg(
            customers=(
                "customer_id",
                "count",
            ),
            avg_probability=(
                "churn_probability",
                "mean",
            ),
            observed_churn=(
                "actual_churn",
                lambda values: (
                    values == "Yes"
                ).mean(),
            ),
        )
    )

    summary["risk_level"] = pd.Categorical(
        summary["risk_level"],
        categories=order,
        ordered=True,
    )

    summary = summary.sort_values(
        "risk_level"
    )

    summary["avg_probability"] = (
        summary["avg_probability"]
        * 100
    ).round(2)

    summary["observed_churn"] = (
        summary["observed_churn"]
        * 100
    ).round(2)

    return summary


def churn_summary(
    data: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    result = (
        data.groupby(
            column,
            as_index=False,
        )
        .agg(
            Clientes=(
                "customerID",
                "count",
            ),
            Churns=(
                "Churn",
                lambda values: (
                    values == "Yes"
                ).sum(),
            ),
        )
    )

    result["Taxa de churn (%)"] = (
        100
        * result["Churns"]
        / result["Clientes"]
    ).round(2)

    return result.sort_values(
        "Taxa de churn (%)",
        ascending=False,
    )


def render_churn_dashboard() -> None:
    (
        data,
        bi,
        final_df,
        selection,
    ) = load_overview_data()

    final = final_df.iloc[0]

    customers = len(data)

    churned = int(
        (data["Churn"] == "Yes").sum()
    )

    churn_rate = (
        churned
        / customers
        * 100
    )

    high_risk = int(
        (bi["risk_level"] == "HIGH").sum()
    )

    selected_model = selection[
        "selected_model"
    ]

    final_auc = float(
        final["roc_auc"]
    )

    st.title(
        "Customer Churn Analytics"
    )

    st.caption(
        "Visão executiva de churn, risco preditivo "
        "e priorização de retenção."
    )

    # ========================================================
    # KPIs
    # ========================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Clientes",
        f"{customers:,}",
    )

    c2.metric(
        "Churn observado",
        f"{churned:,}",
    )

    c3.metric(
        "Taxa de churn",
        f"{churn_rate:.2f}%",
    )

    c4.metric(
        "Clientes de alto risco",
        f"{high_risk:,}",
    )

    c5.metric(
        "ROC-AUC final",
        f"{final_auc:.4f}",
    )

    st.caption(
        "Os indicadores de risco se referem aos "
        f"{len(bi):,} clientes do conjunto de teste final."
    )

    st.divider()

    # ========================================================
    # Executive interpretation
    # ========================================================

    st.subheader(
        "Leitura executiva"
    )

    high = bi[
        bi["risk_level"] == "HIGH"
    ]

    low = bi[
        bi["risk_level"] == "LOW"
    ]

    high_observed = (
        (high["actual_churn"] == "Yes").mean()
        * 100
    )

    low_observed = (
        (low["actual_churn"] == "Yes").mean()
        * 100
    )

    with st.container(border=True):
        st.markdown(
            f"""
### O que os dados mostram?

O projeto analisou **{customers:,} clientes** e encontrou uma
taxa histórica de churn de **{churn_rate:.2f}%**.

Na amostra de teste final:

- **{high_risk} clientes** foram classificados como alto risco;
- o grupo **HIGH** apresentou churn observado de
  **{high_observed:.2f}%**;
- o grupo **LOW** apresentou churn observado de
  **{low_observed:.2f}%**.

O modelo selecionado foi **{selected_model}**, com
**ROC-AUC final de {final_auc:.4f}**.
"""
        )

        st.caption(
            "As diferenças entre os grupos mostram associação "
            "com o risco preditivo. Elas não demonstram que uma "
            "característica específica causa churn."
        )

    st.divider()

    # ========================================================
    # Risk intelligence
    # ========================================================

    st.subheader(
        "Segmentação de risco"
    )

    risk = build_risk_summary(
        bi
    )

    display_risk = risk.rename(
        columns={
            "risk_level": "Risco",
            "retention_priority": "Prioridade",
            "customers": "Clientes",
            "avg_probability": (
                "Probabilidade média (%)"
            ),
            "observed_churn": (
                "Churn observado (%)"
            ),
        }
    )

    left, right = st.columns(
        [3, 2]
    )

    with left:
        st.dataframe(
            display_risk,
            hide_index=True,
            width="stretch",
        )

    with right:
        risk_chart = (
            risk[
                [
                    "risk_level",
                    "customers",
                ]
            ]
            .set_index(
                "risk_level"
            )
            .rename(
                columns={
                    "customers": "Clientes",
                }
            )
        )

        st.bar_chart(
            risk_chart
        )

    st.caption(
        "HIGH, MEDIUM e LOW são faixas operacionais "
        "definidas sobre a probabilidade estimada pelo modelo."
    )

    st.divider()

    # ========================================================
    # Business patterns
    # ========================================================

    st.subheader(
        "Padrões observados nos dados"
    )

    st.caption(
        "Estas análises descrevem o histórico do dataset. "
        "Não significam causalidade."
    )

    contract = churn_summary(
        data,
        "Contract",
    )

    internet = churn_summary(
        data,
        "InternetService",
    )

    left, right = st.columns(2)

    with left:
        st.markdown(
            "#### Churn por tipo de contrato"
        )

        st.bar_chart(
            contract
            .set_index("Contract")[
                "Taxa de churn (%)"
            ]
        )

        st.dataframe(
            contract,
            hide_index=True,
            width="stretch",
        )

    with right:
        st.markdown(
            "#### Churn por serviço de internet"
        )

        st.bar_chart(
            internet
            .set_index("InternetService")[
                "Taxa de churn (%)"
            ]
        )

        st.dataframe(
            internet,
            hide_index=True,
            width="stretch",
        )

    st.divider()

    # ========================================================
    # Model snapshot
    # ========================================================

    st.subheader(
        "Resumo do Machine Learning"
    )

    cv_auc = float(
        selection[
            "selected_cv_roc_auc"
        ]
    )

    with st.container(border=True):
        st.markdown(
            f"""
### {selected_model}

O modelo foi selecionado usando
**5-Fold Stratified Cross-Validation** sobre os
**{selection["development_rows"]:,} clientes de desenvolvimento**.

**ROC-AUC médio na cross-validation:** {cv_auc:.4f}

Depois da seleção, o modelo foi avaliado em
**{selection["final_test_rows"]:,} clientes de teste final**.

**ROC-AUC final:** {final_auc:.4f}

Os detalhes de Precision, Recall, F1, matriz de confusão,
threshold e explicabilidade estão disponíveis na página
**Modelos e Explicabilidade**.
"""
        )

    st.divider()

    # ========================================================
    # Navigation guidance
    # ========================================================

    st.subheader(
        "Explore o projeto"
    )

    nav1, nav2, nav3 = st.columns(3)

    with nav1, st.container(border=True):
        st.markdown(
            "### 🔮 Prever Novo Cliente"
        )
        st.write(
            "Informe os dados de um cliente e "
            "obtenha uma estimativa individual "
            "de risco de churn."
        )

    with nav2, st.container(border=True):
        st.markdown(
            "### 👥 Fila de Retenção"
        )
        st.write(
            "Veja quais clientes estão priorizados "
            "para análise de retenção."
        )

    with nav3, st.container(border=True):
        st.markdown(
            "### 📊 Modelos e Explicabilidade"
        )
        st.write(
            "Analise a metodologia, métricas, "
            "cross-validation e explicabilidade."
        )

    st.caption(
        "Customer Churn Analytics · V1.1"
    )