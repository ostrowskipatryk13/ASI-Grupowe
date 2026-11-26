"""Eksperymenty Sprintu 3: tracking W&B i pomiary modeli"""

import hashlib
import json
import pickle
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd


def git_sha() -> str:
    """SHA bieżącego commita; z dopiskiem -dirty, gdy są niezacommitowane zmiany."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def data_fingerprint(path: str | Path) -> str:
    """SHA-256 danych: z fingerprint.json (pełne dane) albo liczony z pliku (próbka CI)."""
    path = Path(path)
    if not path.is_file():
        return "unknown"
    if path.suffix == ".json":
        return str(json.loads(path.read_text(encoding="utf-8"))["sha256"])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_size_bytes(path: str | Path) -> int:
    """Rozmiar pliku albo katalogu (z podkatalogami) w bajtach."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


# Skrypt uruchamiany w osobnym, czystym procesie: mierzy pamięć potrzebną do wczytania
# modelu i wykonania pierwszej predykcji (AutoGluon wczytuje modele leniwie, przy predict).
_MEMORY_SCRIPT = """
import pickle, sys
import pandas as pd
import psutil
kind, model_path, sample_path = sys.argv[1:4]
X = pd.read_pickle(sample_path)
if kind == "autogluon":
    from autogluon.tabular import TabularPredictor
else:
    import sklearn.pipeline
process = psutil.Process()
before = process.memory_info().rss
if kind == "autogluon":
    model = TabularPredictor.load(model_path)
else:
    with open(model_path, "rb") as handle:
        model = pickle.load(handle)
model.predict(X)
print((process.memory_info().rss - before) / 1_000_000)
"""


def load_model(kind: str, path: str | Path) -> Any:
    """Wczytuje model zapisany jako pickle (sklearn) albo katalog AutoGluona."""
    if kind == "autogluon":
        from autogluon.tabular import TabularPredictor

        return TabularPredictor.load(str(path))
    with open(path, "rb") as handle:
        return pickle.load(handle)


def measure_load_memory_mb(kind: str, path: str | Path, X: pd.DataFrame) -> float:
    """Przyrost pamięci procesu po wczytaniu modelu i pierwszej predykcji (czysty proces)."""
    with tempfile.TemporaryDirectory() as tmp:
        sample_path = Path(tmp) / "sample.pkl"
        X.to_pickle(sample_path)
        result = subprocess.run(
            [sys.executable, "-c", _MEMORY_SCRIPT, kind, str(path), str(sample_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    return float(result.stdout.strip().splitlines()[-1])


def measure_model(kind: str, path: str | Path, X: pd.DataFrame, repeats: int = 5) -> dict:
    """Pomiary modelu: rozmiar artefaktu, czas wczytania, czas predykcji, pamięć."""
    start = time.perf_counter()
    model = load_model(kind, path)
    model.predict(X.iloc[:1])  # pierwsza predykcja: dociąga leniwie wczytywane modele
    load_seconds = time.perf_counter() - start

    one_row = []
    batch = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(X.iloc[:1])
        one_row.append(time.perf_counter() - start)
        start = time.perf_counter()
        model.predict(X)
        batch.append(time.perf_counter() - start)

    return {
        "size_mb": directory_size_bytes(path) / 1_000_000,
        "load_memory_mb": measure_load_memory_mb(kind, path, X),
        "load_time_s": load_seconds,
        "predict_ms_one_row": statistics.median(one_row) * 1000,
        "predict_ms_per_row_batch": statistics.median(batch) * 1000 / len(X),
    }


def log_wandb_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    measurements: dict[str, float],
    wandb_config: dict[str, Any],
) -> dict[str, Any]:
    """Jeden run W&B: parametry, metryki, pomiary, Git SHA i fingerprint danych."""
    import wandb

    config = {
        **params,
        "git_sha": git_sha(),
        "data_fingerprint": data_fingerprint(wandb_config["fingerprint_file"]),
    }
    run = wandb.init(
        project=wandb_config["project"],
        entity=wandb_config.get("entity"),
        name=run_name,
        job_type="experiment",
        config=config,
        reinit="finish_previous",
    )
    scalar_metrics = {key: value for key, value in metrics.items() if key != "confusion_matrix"}
    matrix = {f"cm_{key}": value for key, value in metrics.get("confusion_matrix", {}).items()}
    run.log({**scalar_metrics, **matrix, **measurements})
    run.finish()
    return {"model": run_name, **scalar_metrics, **measurements}
