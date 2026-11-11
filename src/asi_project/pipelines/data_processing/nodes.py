import pandas as pd

from asi_project.config import get_contract, load_project_config
from asi_project.data import (
    assert_valid,
    filter_training_rows,
    normalize_input,
)


def validate_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    config = load_project_config()
    contract = get_contract(config)

    assert_valid(raw_data, contract, config)

    return raw_data


def normalize_data(validated_data: pd.DataFrame) -> pd.DataFrame:
    return normalize_input(validated_data)


def filter_data(normalized_data: pd.DataFrame) -> pd.DataFrame:
    config = load_project_config()
    contract = get_contract(config)

    return filter_training_rows(normalized_data, contract)
