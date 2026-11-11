from sklearn.pipeline import Pipeline

from asi_project.config import get_contract, load_project_config
from asi_project.data import filter_training_rows, load_dataset, normalize_input
from asi_project.pipelines.data_science.nodes import (
    evaluate_model,
    split_data,
    train_model,
)


def test_split_data():
    config = load_project_config()
    contract = get_contract(config)

    raw_data = load_dataset(config, contract, mode="sample")
    prepared = filter_training_rows(normalize_input(raw_data), contract)

    X_train, X_test, y_train, y_test = split_data(prepared)

    assert len(X_train) + len(X_test) == len(prepared)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert y_train.nunique() == 2
    assert y_test.nunique() == 2


def test_train_and_evaluate_model():
    config = load_project_config()
    contract = get_contract(config)

    raw_data = load_dataset(config, contract, mode="sample")
    prepared = filter_training_rows(normalize_input(raw_data), contract)

    X_train, X_test, y_train, y_test = split_data(prepared)
    model = train_model(X_train, y_train)

    assert isinstance(model, Pipeline)

    metrics = evaluate_model(model, X_test, y_test)

    for metric in ("accuracy", "precision", "recall", "f1"):
        assert metric in metrics
        assert 0 <= metrics[metric] <= 1

    confusion = metrics["confusion_matrix"]

    assert set(confusion) == {"tn", "fp", "fn", "tp"}
    assert sum(confusion.values()) == len(y_test)
