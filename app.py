import streamlit as st

from customer_churn.dashboard.about_view import (
    render_about_page,
)
from customer_churn.dashboard.churn_view import (
    render_churn_dashboard,
)
from customer_churn.dashboard.models_view import (
    render_models_page,
)
from customer_churn.dashboard.predict_view import (
    render_predict_customer,
)
from customer_churn.dashboard.retention_view import (
    render_retention_queue,
)

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
)


def render_overview() -> None:
    st.title("Customer Churn Analytics")

    st.caption(
        "Produto independente · versão 1.1"
    )

    st.markdown(
        """
        <style>
        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin: 1.6rem 0 .8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    render_churn_dashboard()


overview_page = st.Page(
    render_overview,
    title="Visão Geral",
    icon="🏠",
    default=True,
)

predict_page = st.Page(
    render_predict_customer,
    title="Prever Novo Cliente",
    icon="🔮",
)

retention_page = st.Page(
    render_retention_queue,
    title="Fila de Retenção",
    icon="👥",
)

models_page = st.Page(
    render_models_page,
    title="Modelos e Explicabilidade",
    icon="📊",
)

about_page = st.Page(
    render_about_page,
    title="Sobre o Projeto",
    icon="📖",
)


navigation = st.navigation(
    {
        "Customer Churn Analytics": [
            overview_page,
            predict_page,
            retention_page,
            models_page,
            about_page,
        ]
    }
)

navigation.run()