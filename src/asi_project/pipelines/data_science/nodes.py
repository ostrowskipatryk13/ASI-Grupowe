import shutil
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from asi_project.config import get_contract, load_project_config
from asi_project.data import make_xy, split_train_test
from asi_project.experiments import data_fingerprint, git_sha, log_wandb_run, measure_model
from asi_project.metrics import compute_metrics
from asi_project.model import build_baseline_pipeline
from asi_project.model_package import load_model, save_model, save_model_package


def split_data(
    prepared_data: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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


# --- Sprint 3: W&B, AutoGluon i porównanie modeli ------------------------------------


def measure_baseline(trained_model: Pipeline, X_test: pd.DataFrame) -> dict:
    """Pomiary baseline'u na kopii zapisanej przez save_model (tak jak trafia na dysk)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pkl"
        save_model(trained_model, "sklearn", path)
        return measure_model("sklearn", path, X_test)


def log_baseline_run(
    model_metrics: dict,
    baseline_measurements: dict,
    test_size: float,
    random_state: int,
    wandb_config: dict,
) -> dict:
    """Pierwszy run W&B: baseline LogisticRegression ze Sprintu 2 (bez zmian)."""
    params = {
        "model": "logistic_regression",
        "test_size": test_size,
        "random_state": random_state,
    }
    return log_wandb_run(
        "baseline_logistic_regression",
        params,
        model_metrics,
        baseline_measurements,
        wandb_config,
    )


def train_autogluon(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    autogluon_params: dict,
    experiment: dict,
    baseline_summary: dict,
) -> str:
    """Trening jednej konfiguracji AutoGluona z limitem czasu; zwraca ścieżkę do modelu."""
    from autogluon.tabular import TabularPredictor

    label = autogluon_params["label"]
    path = Path(autogluon_params["models_dir"]) / experiment["name"]
    shutil.rmtree(path, ignore_errors=True)
    predictor = TabularPredictor(
        label=label,
        problem_type="binary",
        eval_metric=autogluon_params["eval_metric"],
        path=str(path),
        verbosity=1,
    )
    predictor.fit(
        train_data=X_train.assign(**{label: y_train.to_numpy()}),
        presets=experiment["presets"],
        time_limit=experiment["time_limit"],
        hyperparameters=experiment.get("hyperparameters"),
        excluded_model_types=autogluon_params.get("excluded_model_types"),
    )
    return str(path)


def evaluate_autogluon(model_path: str, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Te same metryki co dla baseline'u, liczone na tym samym zbiorze testowym."""
    predictor = load_model("autogluon", model_path)
    predictions = predictor.predict(X_test)
    return compute_metrics(y_test, predictions.astype(int))


def measure_autogluon(model_path: str, X_test: pd.DataFrame) -> dict:
    """Pomiary AutoGluona: rozmiar katalogu, pamięć po wczytaniu, czas predykcji."""
    return measure_model("autogluon", model_path, X_test)


def log_autogluon_run(
    metrics: dict,
    measurements: dict,
    experiment: dict,
    test_size: float,
    random_state: int,
    wandb_config: dict,
) -> dict:
    """Run W&B jednej konfiguracji AutoGluona."""
    params = {
        "model": "autogluon",
        "presets": experiment["presets"],
        "time_limit": experiment["time_limit"],
        "test_size": test_size,
        "random_state": random_state,
    }
    return log_wandb_run(experiment["name"], params, metrics, measurements, wandb_config)


def build_comparison_table(*summaries: dict) -> pd.DataFrame:
    """Tabela porównawcza modeli: metryka główna, czas predykcji, rozmiar, pamięć."""
    columns = [
        "model",
        "f1",
        "precision",
        "recall",
        "predict_ms_one_row",
        "predict_ms_per_row_batch",
        "size_mb",
        "load_memory_mb",
        "load_time_s",
    ]
    return pd.DataFrame(list(summaries))[columns]


def log_comparison_table(model_comparison: pd.DataFrame, wandb_config: dict) -> None:
    """Tabela porównawcza jako osobny run W&B (widok zbiorczy)."""
    import wandb

    run = wandb.init(
        project=wandb_config["project"],
        entity=wandb_config.get("entity"),
        name="model_comparison",
        job_type="comparison",
        reinit="finish_previous",
    )
    run.log({"model_comparison": wandb.Table(dataframe=model_comparison)})
    run.finish()


def build_model_package(
    model_path: str,
    metrics: dict,
    measurements: dict,
    experiment: dict,
    wandb_config: dict,
    package_params: dict,
) -> str:
    config = load_project_config()
    contract = get_contract(config)
    metadata = {
        "model_name": experiment["name"],
        "features": list(config.model_features),
        "target": contract.target,
        "positive_class": contract.positive_class,
        "metrics": metrics,
        "measurements": measurements,
        "training_params": {
            "presets": experiment["presets"],
            "time_limit": experiment["time_limit"],
        },
        "git_sha": git_sha(),
        "data_fingerprint": data_fingerprint(wandb_config["fingerprint_file"]),
    }
    package_dir = save_model_package(
        model_path, package_params["output_dir"], "autogluon", metadata
    )
    return str(package_dir)
