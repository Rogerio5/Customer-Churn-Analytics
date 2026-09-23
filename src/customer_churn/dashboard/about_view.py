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
def load_about_data():
    data = pd.read_csv(DATA_PATH)
    bi = pd.read_csv(BI_PATH)
    final = pd.read_csv(FINAL_PATH)

    with SELECTION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        selection = json.load(file)

    return data, bi, final, selection


def render_simple_view(
    data,
    bi,
    final,
    selection,
):
    customers = len(data)
    churned = int((data["Churn"] == "Yes").sum())
    churn_rate = churned / customers * 100

    low = int((bi["risk_level"] == "LOW").sum())
    medium = int((bi["risk_level"] == "MEDIUM").sum())
    high = int((bi["risk_level"] == "HIGH").sum())

    final_row = final.iloc[0]
    final_auc = float(final_row["roc_auc"])

    # --------------------------------------------------------
    # Resumo do projeto
    # --------------------------------------------------------

    p1, p2, p3, p4 = st.columns(
        4,
        gap="small",
    )

    with p1, st.container(border=True):
        st.markdown("##### 🎯 Problema")
        st.caption(
            "Identificar clientes com maior risco "
            "estimado de churn."
        )

    with p2, st.container(border=True):
        st.markdown("##### 🗂️ Dados")
        st.caption(
            f"{customers:,} clientes com perfil, "
            "serviços e cobrança."
        )

    with p3, st.container(border=True):
        st.markdown("##### 🤖 Modelo")
        st.caption(
            "Logistic Regression selecionada "
            "por cross-validation."
        )

    with p4, st.container(border=True):
        st.markdown("##### 👥 Ação")
        st.caption(
            "Priorizar clientes para retenção."
        )

    # --------------------------------------------------------
    # KPIs compactos
    # --------------------------------------------------------

    k1, k2, k3, k4 = st.columns(
        4,
        gap="small",
    )

    k1.metric(
        "Clientes",
        f"{customers:,}".replace(",", "."),
    )

    k2.metric(
        "Churn",
        f"{churned:,}".replace(",", "."),
    )

    k3.metric(
        "Taxa de churn",
        f"{churn_rate:.2f}%".replace(".", ","),
    )

    k4.metric(
        "ROC-AUC",
        f"{final_auc:.4f}",
    )

    # --------------------------------------------------------
    # Abas compactas
    # --------------------------------------------------------

    flow_tab, risk_tab, navigation_tab = st.tabs(
        [
            "🔄 Como funciona",
            "🚦 Risco",
            "🧭 Navegação",
        ]
    )

    with flow_tab:
        s1, s2, s3, s4 = st.columns(
            4,
            gap="small",
        )

        with s1, st.container(border=True):
            st.markdown("##### 1️⃣ Dados")
            st.caption("Perfil e serviços")

        with s2, st.container(border=True):
            st.markdown("##### 2️⃣ Modelo")
            st.caption("Machine Learning")

        with s3, st.container(border=True):
            st.markdown("##### 3️⃣ Risco")
            st.caption("LOW · MEDIUM · HIGH")

        with s4, st.container(border=True):
            st.markdown("##### 4️⃣ Prioridade")
            st.caption("MONITOR · ENGAGE · URGENT")

        st.caption(
            "A previsão estima risco; não determina "
            "que o cliente irá cancelar."
        )

    with risk_tab:
        r1, r2, r3 = st.columns(
            3,
            gap="small",
        )

        with r1, st.container(border=True):
            st.markdown("##### 🟢 LOW")
            st.metric(
                "Clientes",
                f"{low:,}".replace(",", "."),
            )
            st.caption("< 30% · MONITOR")

        with r2, st.container(border=True):
            st.markdown("##### 🟠 MEDIUM")
            st.metric(
                "Clientes",
                f"{medium:,}".replace(",", "."),
            )
            st.caption("30% a < 60% · ENGAGE")

        with r3, st.container(border=True):
            st.markdown("##### 🔴 HIGH")
            st.metric(
                "Clientes",
                f"{high:,}".replace(",", "."),
            )
            st.caption("≥ 60% · URGENT")

    with navigation_tab:
        n1, n2, n3, n4 = st.columns(
            4,
            gap="small",
        )

        with n1, st.container(border=True):
            st.markdown("##### 🏠 Visão Geral")
            st.caption("KPIs e risco")

        with n2, st.container(border=True):
            st.markdown("##### 🔮 Previsão")
            st.caption("Novo cliente")

        with n3, st.container(border=True):
            st.markdown("##### 👥 Retenção")
            st.caption("Fila operacional")

        with n4, st.container(border=True):
            st.markdown("##### 📊 Modelos")
            st.caption("Métricas e explicação")

def render_technical_view(
    data,
    final,
    selection,
):
    final_row = final.iloc[0]
    selected_model = selection["selected_model"]

    data_tab, validation_tab, explain_tab, stack_tab = st.tabs(
        [
            "🗂️ Dados e preparação",
            "🔁 Validação",
            "🧠 Explicabilidade",
            "🛠️ Tecnologias e limites",
        ]
    )

    with data_tab:
        left, right = st.columns(2, gap="large")

        with left, st.container(border=True):
            st.markdown("#### Dataset")
            st.markdown(
                f"""
- **{len(data):,} clientes**
- Variável alvo: **Churn**
- Perfil e características cadastrais
- Serviços contratados
- Contrato e forma de pagamento
- Cobrança mensal e total
"""
            )

        with right, st.container(border=True):
            st.markdown("#### Pré-processamento")
            st.markdown(
                """
- Numéricas → `StandardScaler`
- Categóricas → `OneHotEncoder`
- Pipeline único de pré-processamento + modelo
- Mesmas transformações no treino e na previsão
"""
            )

    with validation_tab:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(
            "Desenvolvimento",
            f'{selection["development_rows"]:,}',
        )
        c2.metric(
            "Teste final",
            f'{selection["final_test_rows"]:,}',
        )
        c3.metric(
            "CV folds",
            f'{selection["cv"]["folds"]}',
        )
        c4.metric(
            "ROC-AUC CV",
            f'{selection["selected_cv_roc_auc"]:.4f}',
        )
        c5.metric(
            "ROC-AUC final",
            f'{final_row["roc_auc"]:.4f}',
        )

        left, right = st.columns(2, gap="large")

        with left, st.container(border=True):
            st.markdown("#### Seleção do modelo")
            st.write(
                "Logistic Regression, Random Forest e XGBoost "
                "foram comparados por ROC-AUC médio em "
                "Stratified 5-Fold Cross-Validation."
            )
            st.success(f"Selecionado: {selected_model}")

        with right, st.container(border=True):
            st.markdown("#### Teste final")
            st.write(
                "O conjunto final foi preservado até depois da "
                "seleção do algoritmo."
            )

    with explain_tab:
        left, right = st.columns(2, gap="large")

        with left, st.container(border=True):
            st.markdown("#### Logistic Regression")
            st.write(
                "Modelo operacional selecionado. Coeficientes e "
                "contribuições individuais ajudam a explicar o score."
            )

        with right, st.container(border=True):
            st.markdown("#### XGBoost + SHAP")
            st.write(
                "Análise complementar de importância preditiva "
                "em modelo baseado em árvores."
            )

        st.info(
            "Explicabilidade mostra como o modelo chegou ao score; "
            "não demonstra causalidade."
        )

    with stack_tab:
        left, right = st.columns(2, gap="large")

        with left, st.container(border=True):
            st.markdown("#### Tecnologias")
            st.markdown(
                """
`Python` · `Pandas` · `SQL` · `scikit-learn` ·
`XGBoost` · `SHAP` · `Streamlit` · `pytest` · `Ruff`
"""
            )

        with right, st.container(border=True):
            st.markdown("#### Limitações")
            st.markdown(
                """
- Dataset de demonstração
- Associação não significa causalidade
- Threshold pode depender do custo de negócio
- Calibração pode ser aprofundada
- Produção exigiria monitoramento e detecção de drift
"""
            )


def render_about_page():
    (
        data,
        bi,
        final,
        selection,
    ) = load_about_data()

    st.markdown(
        '<div class="churn-eyebrow">Contexto e metodologia</div>',
        unsafe_allow_html=True,
    )
    st.title("Sobre o Projeto")

    st.caption(
        "Problema de negócio, fluxo da solução e metodologia "
        "organizados para leitura rápida."
    )

    simple_tab, technical_tab = st.tabs(
        [
            "👤 Entenda o projeto",
            "🧪 Metodologia",
        ]
    )

    with simple_tab:
        render_simple_view(
            data=data,
            bi=bi,
            final=final,
            selection=selection,
        )

    with technical_tab:
        render_technical_view(
            data=data,
            final=final,
            selection=selection,
        )

    st.caption("Customer Churn Analytics · v1.2.0")
