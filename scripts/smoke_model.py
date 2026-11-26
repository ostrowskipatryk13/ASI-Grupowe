import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml

from asi_project.config import get_contract, load_project_config
from asi_project.data import load_dataset, normalize_input
from asi_project.model_package import load_model_package

N_ROWS = 5
PARAMETERS = Path("conf/base/parameters.yml")


def artifact_path(alias: str) -> str:
    params = yaml.safe_load(PARAMETERS.read_text(encoding="utf-8"))
    wandb_params = params["wandb"]
    name = params["model_package"]["artifact_name"]
    return f"{wandb_params['entity']}/{wandb_params['project']}/{name}:{alias}"


def check_predictions(predictions: list, n_rows: int) -> list[str]:
    problems = []
    if len(predictions) != n_rows:
        problems.append(f"oczekiwano {n_rows} predykcji, jest {len(predictions)}")
    unexpected = sorted(set(predictions) - {0, 1})
    if unexpected:
        problems.append(f"nieoczekiwane wartości predykcji: {unexpected}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test modelu produkcyjnego z W&B.")
    parser.add_argument("--alias", default="production")
    parser.add_argument("--download-dir", default="data/06_models/smoke")
    args = parser.parse_args(argv)

    import wandb

    path = artifact_path(args.alias)
    artifact = wandb.Api().artifact(path)
    download_dir = Path(args.download_dir)
    shutil.rmtree(download_dir, ignore_errors=True)
    artifact.download(root=str(download_dir))

    model, metadata = load_model_package(download_dir)

    config = load_project_config()
    contract = get_contract(config)
    sample = normalize_input(load_dataset(config, contract, mode="sample"))
    missing = [name for name in metadata["features"] if name not in sample.columns]
    if missing:
        print(f"SMOKE TEST NIE PRZESZEDŁ: w danych brakuje cech {missing}")
        return 1
    X = sample[metadata["features"]].head(N_ROWS)
    predictions = pd.Series(model.predict(X)).tolist()

    print(f"Artefakt: {path} ({artifact.version})")
    print(f"Model: {metadata['model_name']}, git {metadata['git_sha'][:7]}")
    print(f"Predykcje dla {N_ROWS} wierszy: {predictions}")
    problems = check_predictions(predictions, N_ROWS)
    if problems:
        print("SMOKE TEST NIE PRZESZEDŁ: " + "; ".join(problems))
        return 1
    print("SMOKE TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
