from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]

BI_PATH = (
    ROOT
    / "data"
    / "processed"
    / "churn_customer_intelligence_bi.csv"
)


@st.cache_data
def load_retention_data() -> pd.DataFrame:
    """Load customer-level retention intelligence."""
    return pd.read_csv(BI_PATH)


def render_retention_queue() -> None:
    st.title("Fila de Retenção")

    st.caption(
        "Priorização operacional dos clientes conforme a "
        "probabilidade estimada de churn."
    )

    st.info(
        "Esta fila organiza os clientes por risco preditivo. "
        "Ela ajuda a decidir quem analisar primeiro, mas não "
        "significa que um cliente certamente irá cancelar."
    )

    data = load_retention_data()

    high = int((data["risk_level"] == "HIGH").sum())
    medium = int((data["risk_level"] == "MEDIUM").sum())
    low = int((data["risk_level"] == "LOW").sum())

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(
            "Clientes avaliados",
            f"{len(data):,}",
        )

    with metric2:
        st.metric(
            "🔴 Alto risco",
            f"{high:,}",
        )

    with metric3:
        st.metric(
            "🟠 Risco moderado",
            f"{medium:,}",
        )

    with metric4:
        st.metric(
            "🟢 Baixo risco",
            f"{low:,}",
        )

    st.divider()

    st.subheader("Como interpretar a fila")

    col1, col2, col3 = st.columns(3)

    with col1, st.container(border=True):
        st.markdown("### 🔴 HIGH")
        st.markdown("**Prioridade: URGENT**")
        st.write(
            "Clientes com probabilidade estimada "
            "de churn a partir de 60%."
        )
        st.caption(
            "Devem aparecer primeiro na análise da "
            "equipe de retenção."
        )

    with col2, st.container(border=True):
        st.markdown("### 🟠 MEDIUM")
        st.markdown("**Prioridade: ENGAGE**")
        st.write(
            "Clientes com probabilidade estimada "
            "entre 30% e menos de 60%."
        )
        st.caption(
            "Podem receber acompanhamento e ações "
            "de relacionamento."
        )

    with col3, st.container(border=True):
        st.markdown("### 🟢 LOW")
        st.markdown("**Prioridade: MONITOR**")
        st.write(
            "Clientes com probabilidade estimada "
            "abaixo de 30%."
        )
        st.caption(
            "Podem permanecer em acompanhamento "
            "normal."
        )

    st.divider()

    st.subheader("Filtrar clientes")

    filter1, filter2, filter3 = st.columns(3)

    with filter1:
        selected_risks = st.multiselect(
            "Nível de risco",
            ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH"],
        )

    with filter2:
        contracts = sorted(
            data["Contract"]
            .dropna()
            .astype(str)
            .unique()
        )

        selected_contracts = st.multiselect(
            "Tipo de contrato",
            contracts,
        )

    with filter3:
        customer_search = st.text_input(
            "Buscar cliente",
            placeholder="Ex.: 2754-SDJRD",
        )

    filtered = data.copy()

    if selected_risks:
        filtered = filtered[
            filtered["risk_level"].isin(
                selected_risks
            )
        ]

    if selected_contracts:
        filtered = filtered[
            filtered["Contract"].isin(
                selected_contracts
            )
        ]

    if customer_search.strip():
        filtered = filtered[
            filtered["customer_id"]
            .astype(str)
            .str.contains(
                customer_search.strip(),
                case=False,
                na=False,
            )
        ]

    filtered = filtered.sort_values(
        "churn_probability",
        ascending=False,
    )

    st.divider()

    st.subheader("Clientes priorizados")

    st.caption(
        "💡 Como ler: a lista começa pelos clientes com "
        "maior probabilidade estimada de churn dentro "
        "dos filtros selecionados."
    )

    count_col, average_col = st.columns(2)

    with count_col:
        st.metric(
            "Clientes encontrados",
            f"{len(filtered):,}",
        )

    with average_col:
        if filtered.empty:
            average_probability = 0.0
        else:
            average_probability = (
                filtered["churn_probability"].mean()
                * 100
            )

        st.metric(
            "Risco médio da seleção",
            f"{average_probability:.2f}%",
        )

    display = filtered[
        [
            "customer_id",
            "churn_probability",
            "risk_level",
            "retention_priority",
            "tenure",
            "MonthlyCharges",
            "Contract",
            "InternetService",
            "PaymentMethod",
            "TechSupport",
            "OnlineSecurity",
            "actual_churn",
        ]
    ].copy()

    display["churn_probability"] = (
        display["churn_probability"]
        .mul(100)
        .map(lambda value: f"{value:.2f}%")
    )

    display = display.rename(
        columns={
            "customer_id": "Cliente",
            "churn_probability": "Probabilidade de churn",
            "risk_level": "Risco",
            "retention_priority": "Prioridade",
            "tenure": "Tempo como cliente (meses)",
            "MonthlyCharges": "Cobrança mensal",
            "Contract": "Contrato",
            "InternetService": "Internet",
            "PaymentMethod": "Pagamento",
            "TechSupport": "Suporte técnico",
            "OnlineSecurity": "Segurança online",
            "actual_churn": "Churn observado",
        }
    )

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "Churn observado indica o resultado existente no "
        "dataset de teste. A probabilidade é a estimativa "
        "produzida pelo modelo de referência."
    )

    csv = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Baixar clientes filtrados",
        data=csv,
        file_name="fila_retencao_filtrada.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()

    with st.expander(
        "Entenda como a prioridade é definida"
    ):
        st.markdown(
            """
A prioridade é uma regra operacional do projeto:

- **HIGH → URGENT:** analisar primeiro.
- **MEDIUM → ENGAGE:** acompanhar e engajar.
- **LOW → MONITOR:** manter monitoramento.

Os limites utilizados são:

- **LOW:** abaixo de 30%.
- **MEDIUM:** de 30% até menos de 60%.
- **HIGH:** a partir de 60%.

Essas faixas ajudam a organizar a operação. Elas não
representam causalidade nem garantem que o cliente irá cancelar.
"""
        )