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




def _format_count(value: int) -> str:
    return f"{int(value):,}".replace(",", ".")


def _format_percent(
    value: float,
    decimals: int = 2,
) -> str:
    return (
        f"{value:.{decimals}f}"
        .replace(".", ",")
        + "%"
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
    churned = int((data["Churn"] == "Yes").sum())
    churn_rate = churned / customers * 100

    high = bi[bi["risk_level"] == "HIGH"]
    medium = bi[bi["risk_level"] == "MEDIUM"]
    low = bi[bi["risk_level"] == "LOW"]

    high_risk = len(high)
    medium_risk = len(medium)
    low_risk = len(low)

    high_observed = (
        (high["actual_churn"] == "Yes").mean()
        * 100
    )

    medium_observed = (
        (medium["actual_churn"] == "Yes").mean()
        * 100
    )

    low_observed = (
        (low["actual_churn"] == "Yes").mean()
        * 100
    )

    selected_model = selection["selected_model"]
    cv_auc = float(selection["selected_cv_roc_auc"])
    final_auc = float(final["roc_auc"])

    risk = (
        build_risk_summary(bi)
        .set_index("risk_level")
    )

    high_probability = float(
        risk.loc["HIGH", "avg_probability"]
    )

    medium_probability = float(
        risk.loc["MEDIUM", "avg_probability"]
    )

    low_probability = float(
        risk.loc["LOW", "avg_probability"]
    )

    # ========================================================
    # IDENTIDADE VISUAL DA VISAO GERAL
    # ========================================================

    st.markdown(
        """
        <style>

        .overview-kicker {
            color: #2563EB;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.10rem;
        }

        .overview-subtitle {
            color: #64748B;
            font-size: 0.92rem;
            margin-top: -0.15rem;
            margin-bottom: 0.20rem;
        }

        .overview-status {
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 0.68rem 0.85rem;
            background: #F8FAFC;
            color: #475569;
            font-size: 0.82rem;
            line-height: 1.45;
        }

        .overview-status strong {
            color: #0F172A;
        }

        .overview-insight {
            border: 1px solid #E2E8F0;
            border-radius: 11px;
            padding: 0.62rem 0.72rem;
            min-height: 76px;
            background: #FFFFFF;
        }

        .overview-insight-label {
            color: #64748B;
            font-size: 0.73rem;
            font-weight: 700;
            margin-bottom: 0.18rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .overview-insight-value {
            color: #0F172A;
            font-size: 1.06rem;
            font-weight: 750;
            line-height: 1.25;
        }

        .overview-insight-note {
            color: #64748B;
            font-size: 0.76rem;
            margin-top: 0.16rem;
        }

        .risk-line {
            display: grid;
            grid-template-columns: 1.05fr 0.72fr 1fr 1fr;
            align-items: center;
            gap: 0.45rem;
            border-bottom: 1px solid #EEF2F7;
            padding: 0.48rem 0.08rem;
        }

        .risk-line:last-child {
            border-bottom: none;
        }

        .risk-name {
            color: #0F172A;
            font-weight: 750;
            font-size: 0.86rem;
        }

        .risk-count {
            color: #0F172A;
            font-weight: 700;
            font-size: 0.88rem;
        }

        .risk-detail {
            color: #64748B;
            font-size: 0.76rem;
        }

        .overview-section-title {
            color: #0F172A;
            font-size: 1.02rem;
            font-weight: 750;
            margin-bottom: 0.35rem;
        }

        div[data-testid="stMetric"] {
            min-height: 76px !important;
            padding: 0.52rem 0.72rem !important;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.77rem !important;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.52rem !important;
        }

        div[data-testid="stVerticalBlock"] {
            gap: 0.72rem;
        }

        div[data-testid="stAlert"] {
            padding-top: 0.50rem !important;
            padding-bottom: 0.50rem !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # CABECALHO EXECUTIVO
    # ========================================================

    header_col, status_col = st.columns(
        [1.55, 0.45],
        gap="medium",
    )

    with header_col:
        st.markdown(
            '<div class="overview-kicker">'
            'Painel executivo · Customer Retention'
            '</div>',
            unsafe_allow_html=True,
        )

        st.title("Customer Churn Analytics")

        st.markdown(
            '<div class="overview-subtitle">'
            'Visão executiva de churn, risco preditivo '
            'e priorização de retenção.'
            '</div>',
            unsafe_allow_html=True,
        )

    with status_col:
        st.markdown(
            f"""
            <div class="overview-status">
                <strong>Modelo operacional</strong><br>
                {selected_model}<br>
                Teste final: {_format_count(len(bi))} clientes
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # KPIs
    # ========================================================

    k1, k2, k3, k4, k5 = st.columns(
        5,
        gap="small",
    )

    k1.metric(
        "Clientes",
        _format_count(customers),
    )

    k2.metric(
        "Churn observado",
        _format_count(churned),
    )

    k3.metric(
        "Taxa de churn",
        _format_percent(churn_rate),
    )

    k4.metric(
        "Alto risco",
        _format_count(high_risk),
    )

    k5.metric(
        "ROC-AUC final",
        f"{final_auc:.4f}",
    )

    # ========================================================
    # PAINEL CENTRAL
    # ========================================================

    executive_col, risk_col = st.columns(
        [1.05, 0.95],
        gap="medium",
    )

    with executive_col, st.container(border=True):
        st.markdown(
            '<div class="overview-section-title">'
            'Leitura executiva'
            '</div>',
            unsafe_allow_html=True,
        )

        i1, i2 = st.columns(
            2,
            gap="small",
        )

        i3, i4 = st.columns(
            2,
            gap="small",
        )

        with i1:
            st.markdown(
                f"""
                <div class="overview-insight">
                    <div class="overview-insight-label">
                        Churn histórico
                    </div>
                    <div class="overview-insight-value">
                        {_format_percent(churn_rate)}
                    </div>
                    <div class="overview-insight-note">
                        {_format_count(churned)} cancelamentos
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with i2:
            st.markdown(
                f"""
                <div class="overview-insight">
                    <div class="overview-insight-label">
                        Prioridade
                    </div>
                    <div class="overview-insight-value">
                        {_format_count(high_risk)} clientes HIGH
                    </div>
                    <div class="overview-insight-note">
                        Análise prioritária de retenção
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with i3:
            st.markdown(
                f"""
                <div class="overview-insight">
                    <div class="overview-insight-label">
                        HIGH observado
                    </div>
                    <div class="overview-insight-value">
                        {_format_percent(high_observed)}
                    </div>
                    <div class="overview-insight-note">
                        Churn observado no grupo
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with i4:
            st.markdown(
                f"""
                <div class="overview-insight">
                    <div class="overview-insight-label">
                        LOW observado
                    </div>
                    <div class="overview-insight-value">
                        {_format_percent(low_observed)}
                    </div>
                    <div class="overview-insight-note">
                        Churn observado no grupo
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "As diferenças entre grupos representam associação "
            "com risco preditivo, não causalidade."
        )

    with risk_col, st.container(border=True):
        st.markdown(
            '<div class="overview-section-title">'
            'Mapa de risco'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="risk-line">
                <div class="risk-name">🔴 HIGH</div>
                <div class="risk-count">
                    {_format_count(high_risk)}
                </div>
                <div class="risk-detail">
                    Prev. {_format_percent(high_probability)}
                </div>
                <div class="risk-detail">
                    Obs. {_format_percent(high_observed)}
                </div>
            </div>

            <div class="risk-line">
                <div class="risk-name">🟠 MEDIUM</div>
                <div class="risk-count">
                    {_format_count(medium_risk)}
                </div>
                <div class="risk-detail">
                    Prev. {_format_percent(medium_probability)}
                </div>
                <div class="risk-detail">
                    Obs. {_format_percent(medium_observed)}
                </div>
            </div>

            <div class="risk-line">
                <div class="risk-name">🟢 LOW</div>
                <div class="risk-count">
                    {_format_count(low_risk)}
                </div>
                <div class="risk-detail">
                    Prev. {_format_percent(low_probability)}
                </div>
                <div class="risk-detail">
                    Obs. {_format_percent(low_observed)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Previsto = probabilidade média · "
            "Observado = churn real no teste final."
        )

    # ========================================================
    # ABAS COMPACTAS
    # ========================================================

    patterns_tab, model_tab, action_tab = st.tabs(
        [
            "📊 Padrões",
            "🤖 Machine Learning",
            "🧭 Operação",
        ]
    )

    # ========================================================
    # PADROES
    # ========================================================

    with patterns_tab:
        contract = churn_summary(
            data,
            "Contract",
        )

        internet = churn_summary(
            data,
            "InternetService",
        )

        contract_top = contract.sort_values(
            "Taxa de churn (%)",
            ascending=False,
        ).iloc[0]

        internet_top = internet.sort_values(
            "Taxa de churn (%)",
            ascending=False,
        ).iloc[0]

        chart1, chart2 = st.columns(
            2,
            gap="medium",
        )

        with chart1:
            st.markdown(
                "##### Contrato"
            )

            st.caption(
                f"Maior taxa: {contract_top['Contract']} · "
                f"{contract_top['Taxa de churn (%)']:.2f}%"
            )

            st.bar_chart(
                contract.set_index(
                    "Contract"
                )[
                    "Taxa de churn (%)"
                ],
                height=180,
            )

        with chart2:
            st.markdown(
                "##### Serviço de internet"
            )

            st.caption(
                "Maior taxa: "
                f"{internet_top['InternetService']} · "
                f"{internet_top['Taxa de churn (%)']:.2f}%"
            )

            st.bar_chart(
                internet.set_index(
                    "InternetService"
                )[
                    "Taxa de churn (%)"
                ],
                height=180,
            )

        with st.expander(
            "Dados dos padrões históricos"
        ):
            t1, t2 = st.columns(
                2,
                gap="small",
            )

            with t1:
                st.dataframe(
                    contract,
                    hide_index=True,
                    width="stretch",
                    height=145,
                )

            with t2:
                st.dataframe(
                    internet,
                    hide_index=True,
                    width="stretch",
                    height=145,
                )

    # ========================================================
    # MACHINE LEARNING
    # ========================================================

    with model_tab:
        m1, m2, m3, m4 = st.columns(
            [1.30, 1, 1, 1],
            gap="small",
        )

        with m1, st.container(border=True):
            st.caption(
                "Modelo selecionado"
            )

            st.markdown(
                f"**{selected_model}**"
            )

        m2.metric(
            "ROC-AUC CV",
            f"{cv_auc:.4f}",
        )

        m3.metric(
            "ROC-AUC final",
            f"{final_auc:.4f}",
        )

        m4.metric(
            "Teste final",
            _format_count(
                selection["final_test_rows"]
            ),
        )

        st.info(
            "Seleção pelo maior ROC-AUC médio em "
            "Stratified 5-Fold Cross-Validation. "
            "O teste final foi preservado até depois "
            "da escolha do modelo."
        )

    # ========================================================
    # OPERACAO
    # ========================================================

    with action_tab:
        a1, a2, a3 = st.columns(
            3,
            gap="small",
        )

        with a1, st.container(border=True):
            st.markdown(
                "##### 🔮 Novo cliente"
            )

            st.caption(
                "Probabilidade, risco e fatores "
                "da previsão."
            )

        with a2, st.container(border=True):
            st.markdown(
                "##### 👥 Retenção"
            )

            st.caption(
                "Priorize clientes e aplique filtros."
            )

        with a3, st.container(border=True):
            st.markdown(
                "##### 📊 Modelos"
            )

            st.caption(
                "Métricas, validação e explicabilidade."
            )

    st.caption(
        "Customer Churn Analytics · v1.2.0"
    )