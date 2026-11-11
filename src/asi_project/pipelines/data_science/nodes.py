import pandas as pd
from sklearn.pipeline import Pipeline

from asi_project.config import get_contract, load_project_config
from asi_project.data import make_xy, split_train_test
from asi_project.metrics import compute_metrics
from asi_project.model import build_baseline_pipeline



def split_data(
    prepared_data: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Dzieli dane zgodnie z kontraktem i parametrami Kedro."""
    config = load_project_config()
    contract = get_contract(config)

    X, y = make_xy(
        prepared_data,
        contract,
        config.model_features,
    )

    return split_train_test(
        X,
        y,
        contract,
        test_size=test_size,
        random_state=random_state,
    )



def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    config = load_project_config()

    model = build_baseline_pipeline(
        config.numeric_features,
        config.categorical_features,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(
    trained_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    predictions = trained_model.predict(X_test)

    return compute_metrics(y_test, predictions)
