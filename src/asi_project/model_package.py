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
    if metadata["model_type"] not in MODEL_FILES:
        raise ValueError(f"{METADATA_FILE}: nieznany typ modelu {metadata['model_type']!r}.")


def save_model_package(
    model: Any,
    package_dir: str | Path,
    model_type: str,
    metadata: dict[str, Any],
) -> Path:
    full_metadata = {
        **metadata,
        "model_type": model_type,
        "model_path": MODEL_FILES.get(model_type),
        "library_versions": library_versions(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    check_metadata(full_metadata)

    package_dir = Path(package_dir)
    shutil.rmtree(package_dir, ignore_errors=True)
    package_dir.mkdir(parents=True)
    model_path = package_dir / full_metadata["model_path"]
    if model_type == "autogluon":
        shutil.copytree(Path(model), model_path)
    else:
        with open(model_path, "wb") as handle:
            pickle.dump(model, handle)
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
    model_path = package_dir / metadata["model_path"]
    if metadata["model_type"] == "autogluon":
        from autogluon.tabular import TabularPredictor

        model = TabularPredictor.load(str(model_path))
    else:
        with open(model_path, "rb") as handle:
            model = pickle.load(handle)
    return model, metadata
