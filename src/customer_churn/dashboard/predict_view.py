from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from customer_churn.analytics.customer_profile import (
    CustomerRetentionProfiler,
)
from customer_churn.data.contract import DEFAULT_CHURN_CONTRACT
from customer_churn.models.logistic_explainability import (
    ChurnLogisticExplainer,
)

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "churn_model.joblib"
)


FEATURE_LABELS = {
    "gender": "Gênero",
    "SeniorCitizen": "Cliente idoso",
    "Partner": "Possui parceiro(a)",
    "Dependents": "Possui dependentes",
    "tenure": "Tempo como cliente",
    "PhoneService": "Serviço telefônico",
    "MultipleLines": "Múltiplas linhas",
    "InternetService": "Serviço de internet",
    "OnlineSecurity": "Segurança online",
    "OnlineBackup": "Backup online",
    "DeviceProtection": "Proteção do dispositivo",
    "TechSupport": "Suporte técnico",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming de filmes",
    "Contract": "Tipo de contrato",
    "PaperlessBilling": "Fatura digital",
    "PaymentMethod": "Forma de pagamento",
    "MonthlyCharges": "Cobrança mensal",
    "TotalCharges": "Total acumulado",
}


@st.cache_resource
def load_churn_model():
    """Load the persisted churn prediction pipeline."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    """Load reference data for valid customer field options."""
    return pd.read_csv(DATA_PATH)


def humanize_feature(feature: str) -> str:
    """Convert encoded feature names into readable labels."""
    categorical_columns = sorted(
        DEFAULT_CHURN_CONTRACT.categorical_columns,
        key=len,
        reverse=True,
    )

    for column in categorical_columns:
        prefix = f"{column}_"

        if feature.startswith(prefix):
            value = feature[len(prefix):]
            label = FEATURE_LABELS.get(column, column)
            return f"{label} = {value}"

    return FEATURE_LABELS.get(feature, feature)


def contribution_dataframe(
    contributions,
    direction: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Build a simple business-friendly contribution table."""
    if direction == "positive":
        selected = [
            item
            for item in contributions
            if item.contribution > 1e-10
        ]

        selected.sort(
            key=lambda item: item.contribution,
            reverse=True,
        )

        effect = "⬆ Aumentou o risco estimado"

    else:
        selected = [
            item
            for item in contributions
            if item.contribution < -1e-10
        ]

        selected.sort(
            key=lambda item: item.contribution,
        )

        effect = "⬇ Reduziu o risco estimado"

    selected = selected[:top_n]

    return pd.DataFrame(
        [
            {
                "Fator identificado": humanize_feature(
                    item.feature
                ),
                "Efeito na previsão": effect,
            }
            for item in selected
        ]
    )

def top_factor_labels(
    contributions,
    direction: str,
    top_n: int,
) -> list[str]:
    """Return the most relevant readable factor labels."""
    if direction == "positive":
        selected = [
            item
            for item in contributions
            if item.contribution > 1e-10
        ]

        selected.sort(
            key=lambda item: item.contribution,
            reverse=True,
        )

    else:
        selected = [
            item
            for item in contributions
            if item.contribution < -1e-10
        ]

        selected.sort(
            key=lambda item: item.contribution,
        )

    return [
        humanize_feature(item.feature)
        for item in selected[:top_n]
    ]


def readable_factor_list(
    factors: list[str],
) -> str:
    """Join factors into a short natural-language list."""
    if not factors:
        return "nenhum fator de maior contribuição"

    if len(factors) == 1:
        return factors[0]

    if len(factors) == 2:
        return f"{factors[0]} e {factors[1]}"

    return (
        ", ".join(factors[:-1])
        + f" e {factors[-1]}"
    )


def operational_guidance(
    risk_level: str,
    priority: str,
) -> str:
    """Return a concise operational interpretation."""
    if risk_level == "HIGH":
        return (
            f"Prioridade {priority}: revisar o cliente "
            "com prioridade na fila de retenção."
        )

    if risk_level == "MEDIUM":
        return (
            f"Prioridade {priority}: acompanhar o cliente "
            "e avaliar ações de engajamento."
        )

    return (
        f"Prioridade {priority}: manter acompanhamento "
        "normal e monitoramento periódico."
    )


def render_predict_customer() -> None:
    st.title("Prever Novo Cliente")

    st.caption(
        "Estimativa de probabilidade de churn para um novo cliente "
        "utilizando o modelo de Machine Learning treinado."
    )

    st.info(
        "A previsão representa uma estimativa de risco baseada nos "
        "padrões do dataset IBM Telco Customer Churn. "
        "Ela não determina que o cliente irá cancelar."
    )

    if not MODEL_PATH.exists():
        st.error(
            "O modelo treinado não foi encontrado."
        )

        st.code(
            "python scripts/train_churn_model.py",
            language="powershell",
        )

        return

    reference = load_reference_data()
    model = load_churn_model()

    st.markdown(
        '<div class="churn-eyebrow">Simulação individual</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Dados do cliente")
    st.caption(
        "Preencha o perfil em quatro grupos e execute a análise."
    )

    col1, col2, col3, col4 = st.columns(
        4,
        gap="medium",
    )

    with col1:
        st.markdown("##### Perfil")
        gender = st.selectbox(
            "Gênero",
            sorted(reference["gender"].dropna().astype(str).unique()),
        )
        senior_citizen = st.selectbox(
            "Cliente idoso",
            [0, 1],
            format_func=lambda value: "Sim" if value == 1 else "Não",
        )
        partner = st.selectbox(
            "Possui parceiro(a)",
            sorted(reference["Partner"].dropna().astype(str).unique()),
        )
        dependents = st.selectbox(
            "Possui dependentes",
            sorted(reference["Dependents"].dropna().astype(str).unique()),
        )
        tenure = st.number_input(
            "Tempo como cliente (meses)",
            min_value=0,
            max_value=int(reference["tenure"].max()),
            value=12,
            step=1,
        )

    with col2:
        st.markdown("##### Telefonia e internet")
        phone_service = st.selectbox(
            "Serviço telefônico",
            sorted(reference["PhoneService"].dropna().astype(str).unique()),
        )
        multiple_lines = st.selectbox(
            "Múltiplas linhas",
            sorted(reference["MultipleLines"].dropna().astype(str).unique()),
        )
        internet_service = st.selectbox(
            "Serviço de internet",
            sorted(reference["InternetService"].dropna().astype(str).unique()),
        )
        online_security = st.selectbox(
            "Segurança online",
            sorted(reference["OnlineSecurity"].dropna().astype(str).unique()),
        )
        online_backup = st.selectbox(
            "Backup online",
            sorted(reference["OnlineBackup"].dropna().astype(str).unique()),
        )

    with col3:
        st.markdown("##### Serviços")
        device_protection = st.selectbox(
            "Proteção do dispositivo",
            sorted(reference["DeviceProtection"].dropna().astype(str).unique()),
        )
        tech_support = st.selectbox(
            "Suporte técnico",
            sorted(reference["TechSupport"].dropna().astype(str).unique()),
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            sorted(reference["StreamingTV"].dropna().astype(str).unique()),
        )
        streaming_movies = st.selectbox(
            "Streaming de filmes",
            sorted(reference["StreamingMovies"].dropna().astype(str).unique()),
        )
        contract = st.selectbox(
            "Tipo de contrato",
            sorted(reference["Contract"].dropna().astype(str).unique()),
        )

    with col4:
        st.markdown("##### Cobrança")
        paperless_billing = st.selectbox(
            "Fatura digital",
            sorted(reference["PaperlessBilling"].dropna().astype(str).unique()),
        )
        payment_method = st.selectbox(
            "Forma de pagamento",
            sorted(reference["PaymentMethod"].dropna().astype(str).unique()),
        )
        monthly_charges = st.number_input(
            "Cobrança mensal",
            min_value=0.0,
            value=float(round(reference["MonthlyCharges"].median(), 2)),
            step=1.0,
            format="%.2f",
        )
        total_charges = st.number_input(
            "Total acumulado de cobranças",
            min_value=0.0,
            value=float(round(reference["TotalCharges"].median(), 2)),
            step=10.0,
            format="%.2f",
        )

    analyze = st.button(
        "Analisar cliente",
        type="primary",
        use_container_width=True,
    )

    if not analyze:
        return

    customer = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )

    probability = float(
        model.predict_proba(customer)[0, 1]
    )

    profile = CustomerRetentionProfiler().build(
        customer_id="NEW-CUSTOMER",
        churn_probability=probability,
    )

    probability_percent = probability * 100

    st.markdown(
        '<div class="churn-eyebrow">Resultado da análise</div>',
        unsafe_allow_html=True,
    )

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Probabilidade de churn",
            f"{probability_percent:.2f}%",
        )

    with result_col2:
        st.metric(
            "Nível de risco",
            profile.risk_level.value,
        )

    with result_col3:
        st.metric(
            "Prioridade",
            profile.retention_priority.value,
        )

    risk_level = profile.risk_level.value

    if risk_level == "HIGH":
        st.error(
            "Alto risco: priorizar a análise deste cliente "
            "na operação de retenção."
        )
    elif risk_level == "MEDIUM":
        st.warning(
            "Risco moderado: acompanhar e avaliar ações "
            "de engajamento."
        )
    else:
        st.success(
            "Baixo risco: manter acompanhamento e "
            "monitoramento periódico."
        )

    explanation = ChurnLogisticExplainer().explain(
        model,
        customer,
    )

    positive_labels = top_factor_labels(
        explanation.contributions,
        direction="positive",
        top_n=3,
    )
    negative_labels = top_factor_labels(
        explanation.contributions,
        direction="negative",
        top_n=2,
    )

    guidance = operational_guidance(
        profile.risk_level.value,
        profile.retention_priority.value,
    )

    risk_plain = {
        "LOW": "baixo",
        "MEDIUM": "moderado",
        "HIGH": "alto",
    }.get(
        profile.risk_level.value,
        profile.risk_level.value,
    )

    positive = contribution_dataframe(
        explanation.contributions,
        direction="positive",
    )
    negative = contribution_dataframe(
        explanation.contributions,
        direction="negative",
    )

    summary_tab, factors_tab, technical_tab = st.tabs(
        [
            "🧭 Resumo operacional",
            "🧠 Fatores da previsão",
            "🔧 Detalhes técnicos",
        ]
    )

    with summary_tab:
        left, right = st.columns(
            [1.05, 0.95],
            gap="large",
        )

        with left, st.container(border=True):
            st.markdown("#### Interpretação")
            st.write(
                f"Este perfil apresenta **risco {risk_plain}**, "
                f"com probabilidade estimada de "
                f"**{probability_percent:.2f}%**."
            )
            st.markdown(f"**Próxima ação:** {guidance}")
            st.caption(
                "A previsão ajuda na priorização, mas não representa "
                "certeza de cancelamento."
            )

        with right, st.container(border=True):
            st.markdown("#### Sinais principais")
            st.markdown("**Aumentaram o risco estimado**")
            if positive_labels:
                for factor in positive_labels:
                    st.markdown(f"- {factor}")
            else:
                st.caption(
                    "Nenhum fator de aumento relevante identificado."
                )

            st.markdown("**Reduziram o risco estimado**")
            if negative_labels:
                for factor in negative_labels:
                    st.markdown(f"- {factor}")
            else:
                st.caption(
                    "Nenhum fator de redução relevante identificado."
                )

    with factors_tab:
        explain_col1, explain_col2 = st.columns(
            2,
            gap="large",
        )

        with explain_col1:
            st.markdown("##### ⬆ Aumentaram o risco")
            if positive.empty:
                st.info(
                    "Nenhuma contribuição positiva relevante."
                )
            else:
                st.dataframe(
                    positive,
                    hide_index=True,
                    width="stretch",
                    height=250,
                )

        with explain_col2:
            st.markdown("##### ⬇ Reduziram o risco")
            if negative.empty:
                st.info(
                    "Nenhuma contribuição negativa relevante."
                )
            else:
                st.dataframe(
                    negative,
                    hide_index=True,
                    width="stretch",
                    height=250,
                )

        st.caption(
            "As contribuições descrevem o comportamento da "
            "Logistic Regression e não representam causalidade."
        )

    with technical_tab:
        technical = pd.DataFrame(
            [
                {
                    "Fator": humanize_feature(item.feature),
                    "Valor transformado": round(
                        item.transformed_value,
                        4,
                    ),
                    "Coeficiente": round(
                        item.coefficient,
                        4,
                    ),
                    "Contribuição": round(
                        item.contribution,
                        4,
                    ),
                }
                for item in explanation.contributions
                if abs(item.contribution) > 1e-10
            ]
        )

        technical = technical.sort_values(
            "Contribuição",
            key=lambda values: values.abs(),
            ascending=False,
        )

        score_col, rule_col = st.columns(
            [0.35, 0.65],
        )
        score_col.metric(
            "Decision score",
            f"{explanation.decision_score:.4f}",
        )
        rule_col.info(
            "LOW < 30% · MEDIUM 30% a < 60% · HIGH ≥ 60%"
        )

        st.dataframe(
            technical,
            hide_index=True,
            width="stretch",
            height=310,
        )

        with st.expander("Ver dados enviados ao modelo"):
            display_customer = customer.copy()
            display_customer.insert(
                0,
                "Probabilidade de churn",
                f"{probability_percent:.2f}%",
            )
            display_customer.insert(
                1,
                "Risco",
                profile.risk_level.value,
            )
            display_customer.insert(
                2,
                "Prioridade",
                profile.retention_priority.value,
            )
            st.dataframe(
                display_customer,
                hide_index=True,
                width="stretch",
            )

    st.caption(
        "Estimativa preditiva baseada nos padrões do dataset "
        "IBM Telco Customer Churn."
    )
