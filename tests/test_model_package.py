import json

import pytest

from asi_project.data import filter_training_rows, make_xy, normalize_input
from asi_project.model import build_baseline_pipeline
from asi_project.model_package import (
    METADATA_FILE,
    REQUIRED_METADATA,
    load_model_package,
    read_metadata,
    save_model_package,
)

N_ROWS = 5


def _training_data(project):
    data = filter_training_rows(normalize_input(project.sample), project.contract)
    return make_xy(data, project.contract, project.config.model_features)


def _metadata(project) -> dict:
    return {
        "model_name": "test_model",
        "features": list(project.config.model_features),
        "target": project.contract.target,
        "positive_class": project.contract.positive_class,
        "metrics": {"f1": 0.5},
        "git_sha": "test",
        "data_fingerprint": "test",
    }


def _check_predictions(predictions, n_rows: int) -> None:
    values = list(predictions)
    assert len(values) == n_rows
    assert set(values) <= {0, 1}


def test_sklearn_package_save_load_predict(project, tmp_path):
    X, y = _training_data(project)
    model = build_baseline_pipeline(
        project.config.numeric_features, project.config.categorical_features
    )
    model.fit(X, y)

    package = save_model_package(model, tmp_path / "package", "sklearn", _metadata(project))
    loaded, metadata = load_model_package(package)

    assert set(REQUIRED_METADATA) <= set(metadata)
    assert metadata["features"] == list(project.config.model_features)
    assert "scikit-learn" in metadata["library_versions"]
    _check_predictions(loaded.predict(X[metadata["features"]].head(N_ROWS)), N_ROWS)


def test_autogluon_package_save_load_predict(project, tmp_path):
    tabular = pytest.importorskip("autogluon.tabular")
    X, y = _training_data(project)
    label = project.contract.target
    predictor = tabular.TabularPredictor(label=label, path=str(tmp_path / "ag"), verbosity=0)
    predictor.fit(
        X.assign(**{label: y.to_numpy()}),
        hyperparameters={"GBM": {"num_boost_round": 10}},
        time_limit=60,
    )

    package = save_model_package(
        tmp_path / "ag", tmp_path / "package", "autogluon", _metadata(project)
    )
    loaded, metadata = load_model_package(package)

    assert metadata["model_type"] == "autogluon"
    assert "autogluon.tabular" in metadata["library_versions"]
    _check_predictions(loaded.predict(X[metadata["features"]].head(N_ROWS)), N_ROWS)


def test_save_rejects_incomplete_metadata(project, tmp_path):
    metadata = _metadata(project)
    del metadata["features"]
    with pytest.raises(ValueError, match="features"):
        save_model_package(object(), tmp_path / "package", "sklearn", metadata)


def test_read_rejects_metadata_without_required_key(project, tmp_path):
    metadata = {
        **_metadata(project),
        "model_type": "sklearn",
        "model_path": "model.pkl",
        "library_versions": {},
    }
    del metadata["git_sha"]
    (tmp_path / METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="git_sha"):
        read_metadata(tmp_path)
