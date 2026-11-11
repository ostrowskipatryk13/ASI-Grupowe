from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DATASETS = ("raw_data", "prepared_data", "trained_model", "model_metrics")


def read_yaml(*parts: str) -> dict:
    path = REPO_ROOT.joinpath(*parts)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@pytest.mark.parametrize("env", ["base", "ci"])
def test_catalog_describes_pipeline_inputs_and_outputs(env):
    catalog = read_yaml("conf", env, "catalog.yml")

    for name in PIPELINE_DATASETS:
        assert name in catalog, f"conf/{env}/catalog.yml: brak {name}"
        assert catalog[name]["type"].endswith("Dataset")


def test_ci_uses_sample_and_separate_outputs():
    base = read_yaml("conf", "base", "catalog.yml")
    ci = read_yaml("conf", "ci", "catalog.yml")

    assert ci["raw_data"]["filepath"] == "data/01_raw/sample.csv"
    for name in PIPELINE_DATASETS[1:]:
        assert ci[name]["filepath"] != base[name]["filepath"]


def test_split_parameters_match_project_yaml():
    params = read_yaml("conf", "base", "parameters.yml")
    project = read_yaml("project.yaml")

    assert params["test_size"] == project["split"]["test_size"]
    assert params["random_state"] == project["split"]["random_state"]
