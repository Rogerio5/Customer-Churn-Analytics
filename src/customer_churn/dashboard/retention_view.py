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
    st.markdown(
        '<div class="churn-eyebrow">Operação de retenção</div>',
        unsafe_allow_html=True,
    )
    st.title("Fila de Retenção")
    st.caption(
        "Priorize clientes por probabilidade estimada de churn, "
        "aplique filtros e exporte a seleção."
    )

    data = load_retention_data()

    high = int((data["risk_level"] == "HIGH").sum())
    medium = int((data["risk_level"] == "MEDIUM").sum())
    low = int((data["risk_level"] == "LOW").sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clientes avaliados", f"{len(data):,}")
    m2.metric("🔴 Alto risco", f"{high:,}")
    m3.metric("🟠 Risco moderado", f"{medium:,}")
    m4.metric("🟢 Baixo risco", f"{low:,}")

    operation_tab, rules_tab = st.tabs(
        ["🎯 Fila operacional", "ℹ️ Como interpretar"]
    )

    with operation_tab:
        f1, f2, f3 = st.columns([1, 1, 1.15], gap="medium")

        with f1:
            selected_risks = st.multiselect(
                "Nível de risco",
                ["HIGH", "MEDIUM", "LOW"],
                default=["HIGH"],
            )

        with f2:
            contracts = sorted(
                data["Contract"].dropna().astype(str).unique()
            )
            selected_contracts = st.multiselect(
                "Tipo de contrato",
                contracts,
            )

        with f3:
            customer_search = st.text_input(
                "Buscar cliente",
                placeholder="Ex.: 2754-SDJRD",
            )

        filtered = data.copy()

        if selected_risks:
            filtered = filtered[
                filtered["risk_level"].isin(selected_risks)
            ]

        if selected_contracts:
            filtered = filtered[
                filtered["Contract"].isin(selected_contracts)
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

        average_probability = (
            0.0
            if filtered.empty
            else filtered["churn_probability"].mean() * 100
        )

        s1, s2, s3 = st.columns([0.8, 0.8, 1.4])
        s1.metric("Clientes encontrados", f"{len(filtered):,}")
        s2.metric("Risco médio", f"{average_probability:.2f}%")

        active_risks = (
            ", ".join(selected_risks)
            if selected_risks
            else "Todos"
        )
        s3.info(
            f"Faixas ativas: {active_risks}. "
            "A tabela começa pelos maiores riscos."
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
                "tenure": "Tempo (meses)",
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
            height=430,
        )

        d1, d2 = st.columns([0.7, 1.3])

        csv = filtered.to_csv(index=False).encode("utf-8")

        with d1:
            st.download_button(
                "Baixar seleção em CSV",
                data=csv,
                file_name="fila_retencao_filtrada.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with d2:
            st.caption(
                "Churn observado é o resultado existente no dataset "
                "de teste; a probabilidade é a estimativa do modelo."
            )

    with rules_tab:
        r1, r2, r3 = st.columns(3, gap="medium")

        with r1, st.container(border=True):
            st.markdown("#### 🔴 HIGH")
            st.markdown("**URGENT · 60% ou mais**")
            st.write("Analisar primeiro na operação de retenção.")

        with r2, st.container(border=True):
            st.markdown("#### 🟠 MEDIUM")
            st.markdown("**ENGAGE · 30% a < 60%**")
            st.write("Acompanhar e avaliar ações de engajamento.")

        with r3, st.container(border=True):
            st.markdown("#### 🟢 LOW")
            st.markdown("**MONITOR · abaixo de 30%**")
            st.write("Manter acompanhamento normal.")

        st.info(
            "As faixas são regras operacionais do projeto. "
            "Elas não representam causalidade e não garantem "
            "que um cliente irá cancelar."
        )
