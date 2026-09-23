import streamlit as st


def apply_ui_style() -> None:
    """Apply the shared visual identity used by the Streamlit app."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1540px;
            padding-top: 1.7rem;
            padding-bottom: 2.5rem;
        }

        h1 {
            letter-spacing: -0.035em;
            margin-bottom: 0.25rem;
        }

        h2, h3 {
            letter-spacing: -0.02em;
        }

        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stMetricLabel"] {
            color: #64748B;
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            color: #0F172A;
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 14px;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid #E5E7EB;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 10px;
            min-height: 2.65rem;
            font-weight: 600;
        }

        .stButton > button[kind="primary"] {
            background: #2563EB;
            border-color: #2563EB;
            color: #FFFFFF;
        }

        .stButton > button[kind="primary"]:hover {
            background: #1D4ED8;
            border-color: #1D4ED8;
            color: #FFFFFF;
        }

        hr {
            margin-top: 1.35rem;
            margin-bottom: 1.35rem;
        }

        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        .churn-eyebrow {
            color: #2563EB;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }

        .churn-note {
            color: #64748B;
            font-size: 0.9rem;
        }

        /* premium-ui-v12 */

        .block-container {
            max-width: 1600px;
            padding-top: 1.15rem;
            padding-bottom: 1.35rem;
        }

        h1 {
            font-size: 2.35rem !important;
            line-height: 1.08 !important;
            margin-top: 0 !important;
            margin-bottom: 0.20rem !important;
        }

        h2 {
            font-size: 1.55rem !important;
            line-height: 1.15 !important;
            margin-top: 0.65rem !important;
            margin-bottom: 0.30rem !important;
        }

        h3 {
            font-size: 1.18rem !important;
            margin-top: 0.45rem !important;
            margin-bottom: 0.25rem !important;
        }

        h4 {
            font-size: 1.02rem !important;
        }

        div[data-testid="stMetric"] {
            padding: 0.70rem 0.85rem;
            min-height: 96px;
            border-radius: 13px;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.82rem;
            line-height: 1.1;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.85rem;
            line-height: 1.05;
        }

        div[data-testid="stAlert"] {
            padding: 0.70rem 0.90rem;
            border-radius: 11px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 13px;
        }

        div[data-baseweb="tab-list"] {
            gap: 1.10rem;
        }

        button[data-baseweb="tab"] {
            padding-top: 0.55rem;
            padding-bottom: 0.55rem;
        }

        .stSelectbox div[data-baseweb="select"] > div {
            min-height: 2.45rem;
        }

        .stNumberInput input,
        .stTextInput input {
            min-height: 2.45rem;
        }

        .stButton > button,
        .stDownloadButton > button {
            min-height: 2.45rem;
        }

        p {
            margin-bottom: 0.45rem;
        }

        .churn-eyebrow {
            margin-bottom: 0.15rem;
        }

        /* about-super-compact-v12 */

        div[data-testid="stMetric"] {
            min-height: 82px;
            padding-top: 0.55rem;
            padding-bottom: 0.55rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.65rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            padding-top: 0.15rem;
            padding-bottom: 0.15rem;
        }

        div[data-testid="stVerticalBlock"] {
            gap: 0.65rem;
        }

        button[data-baseweb="tab"] {
            padding-top: 0.35rem;
            padding-bottom: 0.35rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
