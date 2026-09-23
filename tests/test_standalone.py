from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from customer_churn.data.cleaner import ChurnDataCleaner
from customer_churn.data.loader import ChurnDatasetLoader

ROOT = Path(__file__).resolve().parents[1]


def test_copied_dataset_matches_cleaning():
    raw = ChurnDatasetLoader().load(ROOT / "data/raw/telco_customer_churn.csv")
    expected = pd.read_csv(ROOT / "data/processed/telco_customer_churn_clean.csv")
    actual = ChurnDataCleaner().clean(raw)
    assert len(raw) == 7043
    pd.testing.assert_frame_equal(actual, expected)


def test_dashboard_overview_matches_current_artifacts():
    bi = pd.read_csv(
        ROOT
        / "data"
        / "processed"
        / "churn_customer_intelligence_bi.csv"
    )

    expected_high = int(
        (bi["risk_level"] == "HIGH").sum()
    )

    app = AppTest.from_file(
        str(ROOT / "app.py")
    ).run(timeout=60)

    assert not app.exception

    assert app.metric[0].value == "7.043"

    assert app.metric[3].value == (
        f"{expected_high:,}"
    )

    assert app.metric[4].value == "0.8481"

