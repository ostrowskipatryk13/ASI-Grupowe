import json
import pickle
import platform
import shutil
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

METADATA_FILE = "metadata.json"
MODEL_FILES = {"autogluon": "model", "sklearn": "model.pkl"}
REQUIRED_METADATA = (
    "model_type",
    "model_path",
    "features",
    "target",
    "positive_class",
    "metrics",
    "git_sha",
    "data_fingerprint",
    "library_versions",
)
LIBRARIES = (
    "autogluon.tabular",
    "autogluon.core",
    "autogluon.common",
    "lightgbm",
    "catboost",
    "scikit-learn",
    "pandas",
    "numpy",
)


def _check_model_type(model_type: str) -> None:
    if model_type not in MODEL_FILES:
        raise ValueError(f"Nieznany typ modelu {model_type!r}; dozwolone: {sorted(MODEL_FILES)}.")


def save_model(model: Any, model_type: str, path: str | Path) -> None:
    _check_model_type(model_type)
    if model_type == "autogluon":
        shutil.copytree(Path(model), Path(path))
    else:
        with open(path, "wb") as handle:
            pickle.dump(model, handle)


def load_model(model_type: str, path: str | Path) -> Any:
    _check_model_type(model_type)
    if model_type == "autogluon":
        from autogluon.tabular import TabularPredictor

        return TabularPredictor.load(str(path))
    with open(path, "rb") as handle:
        return pickle.load(handle)


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def check_metadata(metadata: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_METADATA if key not in metadata]
    if missing:
        raise ValueError(f"{METADATA_FILE}: brak wymaganych kluczy {missing}.")
    _check_model_type(metadata["model_type"])


def save_model_package(
    model: Any,
    package_dir: str | Path,
    model_type: str,
    metadata: dict[str, Any],
) -> Path:

    _check_model_type(model_type)
    full_metadata = {
        **metadata,
        "model_type": model_type,
        "model_path": MODEL_FILES[model_type],
        "library_versions": library_versions(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    check_metadata(full_metadata)

    package_dir = Path(package_dir)
    shutil.rmtree(package_dir, ignore_errors=True)
    package_dir.mkdir(parents=True)
    save_model(model, model_type, package_dir / MODEL_FILES[model_type])
    (package_dir / METADATA_FILE).write_text(
        json.dumps(full_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return package_dir


def read_metadata(package_dir: str | Path) -> dict[str, Any]:
    path = Path(package_dir) / METADATA_FILE
    metadata = json.loads(path.read_text(encoding="utf-8"))
    check_metadata(metadata)
    return metadata


def load_model_package(package_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    package_dir = Path(package_dir)
    metadata = read_metadata(package_dir)
    model = load_model(metadata["model_type"], package_dir / metadata["model_path"])
    return model, metadata
