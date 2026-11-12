from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asi_project.data import DataContractError
from asi_project.pipelines.data_processing.nodes import (
    filter_data,
    normalize_data,
    validate_data,
)

SAMPLE_PATH = Path(__file__).resolve().parents[3] / "data" / "01_raw" / "sample.csv"


@pytest.fixture
def sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


def test_validate_data_accepts_sample(sample):
    result = validate_data(sample)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample)


def test_validate_data_fixes_double_spaces_in_column_names(sample):
    raw = sample.rename(columns={"Call Failure": "Call  Failure"})

    result = validate_data(raw)

    assert "Call Failure" in result.columns


def test_validate_data_rejects_missing_target(sample):
    with pytest.raises(DataContractError):
        validate_data(sample.drop(columns=["Churn"]))


def test_normalize_data_keeps_rows_and_is_idempotent(sample):
    once = normalize_data(sample)
    twice = normalize_data(once)

    assert len(once) == len(sample)
    pd.testing.assert_frame_equal(once, twice)


def test_filter_data_removes_rows_without_target(sample):
    df = sample.copy()
    df["Churn"] = df["Churn"].astype(float)
    df.loc[df.index[:3], "Churn"] = np.nan

    result = filter_data(df)

    assert len(result) == len(df) - 3
    assert result["Churn"].notna().all()
