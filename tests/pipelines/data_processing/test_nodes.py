import pandas as pd

from asi_project.pipelines.data_processing.nodes import (
    filter_data,
    normalize_data,
    validate_data,
)


def test_validate_data():
    df = pd.read_csv("data/01_raw/sample.csv")

    result = validate_data(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)


def test_normalize_data():
    df = pd.read_csv("data/01_raw/sample.csv")

    result = normalize_data(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)


def test_filter_data():
    df = pd.read_csv("data/01_raw/sample.csv")

    result = filter_data(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) <= len(df)
