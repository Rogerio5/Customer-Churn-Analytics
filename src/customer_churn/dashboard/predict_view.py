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

    st.subheader("Dados do cliente")

    st.write(
        "Preencha as informações abaixo e clique em "
        "**Analisar cliente**."
    )

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gênero",
            sorted(
                reference["gender"]
                .dropna()
                .astype(str)
                .unique()
            ),
        )

        senior_citizen = st.selectbox(
            "Cliente idoso",
            [0, 1],
            format_func=lambda value: (
                "Sim" if value == 1 else "Não"
            ),
        )

        partner = st.selectbox(
            "Possui parceiro(a)",
            sorted(
                reference["Partner"]
                .dropna()
                .astype(str)
                .unique()
            ),
        )

        dependents = st.selectbox(
            "Possui dependentes",
            sorted(
                reference["Dependents"]
                .dropna()
                .astype(str)
                .unique()
            ),
        )

        tenure = st.number_input(
            "Tempo como cliente (meses)",
            min_value=0,
            max_value=int(reference["tenure"].max()),
            value=12,
            step=1,
        )

        phone_service = st.selectbox(
            "Serviço telefônico",
            sorted(
                reference["PhoneService"]
                .dropna()
                .astype(str)
                .unique()
            ),
        )

        multiple_lines = st.selectbox(
            "Múltiplas linhas",
            sorted(
                reference["MultipleLines"]
                .dropna()
                .astype(str)
                .unique()
            ),
        )

        internet_service = st.selectbox(
            "Serviço de internet",
            sorted(
                reference["InternetService"]
                .dropna()
                .astype(str)
                .unique()
            ),
        )

        online_security = st.selectbox(
            "Segurança online",
            sorted(
                reference["OnlineSecurity"]
                .dropna()
                .astype(str)
                .unique()
            ),
        )

        online_backup = st.selectbox(
            "Backup online",
            sorted(
                reference["OnlineBackup"]
                .dropna()
                .astype(str)
                .unique()
            ),
        )

    with col2:
        device_protection = st.selectbox(
            "Proteção do dispositivo",
            sorted(
                reference["DeviceProtection"]
                .dropna()
                .astype(str)
                .unique()
            ),
        )

        tech_support = st.selectbox(
            "Suporte técnico",
            sorted(
                reference["TechSupport"]
                .dropna()
                .astype(str)
                .unique()
            ),
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            sorted(
                reference["StreamingTV"]
                .dropna()
                .astype(str)
                .unique()
            ),
        )

        streaming_movies = st.selectbox(
            "Streaming de filmes",
            sorted(
                reference["StreamingMovies"]
                .dropna()
                .astype(str)
                .unique()
            ),
        )

        contract = st.selectbox(
            "Tipo de contrato",
            sorted(
                reference["Contract"]
                .dropna()
                .astype(str)
                .unique()
            ),
        )

        paperless_billing = st.selectbox(
            "Fatura digital",
            sorted(
                reference["PaperlessBilling"]
                .dropna()
                .astype(str)
                .unique()
            ),
        )

        payment_method = st.selectbox(
            "Forma de pagamento",
            sorted(
                reference["PaymentMethod"]
                .dropna()
                .astype(str)
                .unique()
            ),
        )

        monthly_charges = st.number_input(
            "Cobrança mensal",
            min_value=0.0,
            value=float(
                round(
                    reference["MonthlyCharges"].median(),
                    2,
                )
            ),
            step=1.0,
            format="%.2f",
        )

        total_charges = st.number_input(
            "Total acumulado de cobranças",
            min_value=0.0,
            value=float(
                round(
                    reference["TotalCharges"].median(),
                    2,
                )
            ),
            step=10.0,
            format="%.2f",
        )

    st.divider()

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

    st.subheader("Resultado da análise")

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
            "Alto risco de churn. "
            "O cliente deve receber prioridade na análise "
            "da equipe de retenção."
        )

    elif risk_level == "MEDIUM":
        st.warning(
            "Risco intermediário de churn. "
            "Recomenda-se acompanhamento e ações de engajamento."
        )

    else:
        st.success(
            "Baixo risco de churn. "
            "O cliente pode permanecer em monitoramento."
        )

    st.caption(
        "Segmentação operacional: "
        "LOW < 30% | MEDIUM de 30% a menos de 60% | "
        "HIGH a partir de 60%."
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

    positive_bullets = "\n".join(
        f"- **{factor}**"
        for factor in positive_labels
    )

    negative_bullets = "\n".join(
        f"- **{factor}**"
        for factor in negative_labels
    )

    if not positive_bullets:
        positive_bullets = (
            "- Nenhum sinal de aumento relevante foi identificado."
        )

    if not negative_bullets:
        negative_bullets = (
            "- Nenhum sinal de redução relevante foi identificado."
        )

    st.markdown("### 🧠 Entenda este resultado")

    with st.container(border=True):
        st.markdown(
            f"""
Este cliente apresenta **risco {risk_plain} de cancelamento**,
com probabilidade estimada de **{probability_percent:.2f}%**.

O modelo encontrou características no perfil que contribuíram
para aumentar ou reduzir essa estimativa.

**Sinais que mais aumentaram o risco estimado:**

{positive_bullets}

**Sinais que ajudaram a reduzir o risco estimado:**

{negative_bullets}

**O que fazer agora:** {guidance}
"""
        )

        st.caption(
            "Esta leitura resume a previsão do próprio modelo. "
            "Ela ajuda a entender o resultado, mas não significa "
            "que esses fatores causaram o cancelamento nem que o "
            "cliente certamente irá cancelar."
        )

    st.divider()

    st.subheader("Por que o modelo chegou a esse resultado?")

    st.caption(
        "A explicação abaixo mostra as contribuições das variáveis "
        "para o score da Logistic Regression. "
        "Valores positivos aumentam o score de churn e valores "
        "negativos reduzem. Essas contribuições não representam causalidade."
    )

    positive = contribution_dataframe(
        explanation.contributions,
        direction="positive",
    )

    negative = contribution_dataframe(
        explanation.contributions,
        direction="negative",
    )

    explain_col1, explain_col2 = st.columns(2)

    with explain_col1:
        st.markdown("#### ⬆ Sinais que aumentaram o risco estimado")

        with st.container(border=True):
            st.caption(
                "💡 Como ler: estes são os sinais do perfil "
                "que mais puxaram a previsão de churn para cima."
            )

        if positive.empty:
            st.info(
                "Nenhuma contribuição positiva relevante "
                "foi identificada."
            )
        else:
            st.dataframe(
                positive,
                hide_index=True,
                width="stretch",
            )

    with explain_col2:
        st.markdown("#### ⬇ Sinais que reduziram o risco estimado")

        with st.container(border=True):
            st.caption(
                "💡 Como ler: estes são os sinais do perfil "
                "que ajudaram a puxar a previsão de churn "
                "para baixo."
            )

        if negative.empty:
            st.info(
                "Nenhuma contribuição negativa relevante "
                "foi identificada."
            )
        else:
            st.dataframe(
                negative,
                hide_index=True,
                width="stretch",
            )

    with st.expander("Detalhes técnicos da explicação"):
        with st.container(border=True):
            st.caption(
                "🔧 Área técnica: destinada a quem deseja entender "
                "o cálculo em maior profundidade. Aqui aparecem os "
                "valores após o pré-processamento, os coeficientes "
                "aprendidos pela Logistic Regression e a contribuição "
                "numérica de cada variável."
            )

        technical = pd.DataFrame(
            [
                {
                    "Fator": humanize_feature(
                        item.feature
                    ),
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

        st.metric(
            "Score linear da Logistic Regression",
            f"{explanation.decision_score:.4f}",
        )

        st.dataframe(
            technical,
            hide_index=True,
            width="stretch",
        )

    st.divider()

    with st.expander("Ver dados analisados"):
        with st.container(border=True):
            st.caption(
                "📥 Dados utilizados: estas são as informações "
                "preenchidas no formulário e enviadas ao modelo "
                "para calcular a previsão apresentada acima."
            )

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

    st.divider()

    st.caption(
        "Esta previsão é uma estimativa preditiva e não uma "
        "conclusão causal sobre o comportamento do cliente."
    )