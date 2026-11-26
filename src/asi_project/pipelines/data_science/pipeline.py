from kedro.pipeline import Pipeline, node

from .nodes import (
    build_comparison_table,
    build_model_package,
    evaluate_autogluon,
    evaluate_model,
    log_autogluon_run,
    log_baseline_run,
    log_comparison_table,
    log_model_package_artifact,
    measure_autogluon,
    measure_baseline,
    split_data,
    train_autogluon,
    train_model,
)

# Nazwy konfiguracji AutoGluona — muszą istnieć w parameters.yml
AUTOGLUON_EXPERIMENTS = ("ag_fast", "ag_quality")
PRODUCTION_MODEL = "ag_fast"


def _autogluon_nodes(name: str) -> list:
    """Cztery nody jednej konfiguracji: trening → ewaluacja → pomiary → run W&B."""
    return [
        node(
            func=train_autogluon,
            inputs=[
                "X_train",
                "y_train",
                "params:autogluon",
                f"params:autogluon.experiments.{name}",
                "baseline_summary",
            ],
            outputs=f"{name}_model_path",
            name=f"train_{name}_node",
        ),
        node(
            func=evaluate_autogluon,
            inputs=[f"{name}_model_path", "X_test", "y_test"],
            outputs=f"{name}_metrics",
            name=f"evaluate_{name}_node",
        ),
        node(
            func=measure_autogluon,
            inputs=[f"{name}_model_path", "X_test"],
            outputs=f"{name}_measurements",
            name=f"measure_{name}_node",
        ),
        node(
            func=log_autogluon_run,
            inputs=[
                f"{name}_metrics",
                f"{name}_measurements",
                f"params:autogluon.experiments.{name}",
                "params:test_size",
                "params:random_state",
                "params:wandb",
            ],
            outputs=f"{name}_summary",
            name=f"log_{name}_run_node",
        ),
    ]


def create_pipeline() -> Pipeline:
    baseline_nodes = [
        node(
            func=split_data,
            inputs=[
                "prepared_data",
                "params:test_size",
                "params:random_state",
            ],
            outputs=[
                "X_train",
                "X_test",
                "y_train",
                "y_test",
            ],
            name="split_data_node",
        ),
        node(
            func=train_model,
            inputs=["X_train", "y_train"],
            outputs="trained_model",
            name="train_model_node",
        ),
        node(
            func=evaluate_model,
            inputs=["trained_model", "X_test", "y_test"],
            outputs="model_metrics",
            name="evaluate_model_node",
        ),
        node(
            func=measure_baseline,
            inputs=["trained_model", "X_test"],
            outputs="baseline_measurements",
            name="measure_baseline_node",
        ),
        node(
            func=log_baseline_run,
            inputs=[
                "model_metrics",
                "baseline_measurements",
                "params:test_size",
                "params:random_state",
                "params:wandb",
            ],
            outputs="baseline_summary",
            name="log_baseline_run_node",
        ),
    ]

    autogluon_nodes = [n for name in AUTOGLUON_EXPERIMENTS for n in _autogluon_nodes(name)]

    comparison_nodes = [
        node(
            func=build_comparison_table,
            inputs=["baseline_summary", *[f"{name}_summary" for name in AUTOGLUON_EXPERIMENTS]],
            outputs="model_comparison",
            name="build_comparison_table_node",
        ),
        node(
            func=log_comparison_table,
            inputs=["model_comparison", "params:wandb"],
            outputs=None,
            name="log_comparison_table_node",
        ),
    ]

    package_nodes = [
        node(
            func=build_model_package,
            inputs=[
                f"{PRODUCTION_MODEL}_model_path",
                f"{PRODUCTION_MODEL}_metrics",
                f"{PRODUCTION_MODEL}_measurements",
                f"params:autogluon.experiments.{PRODUCTION_MODEL}",
                "params:wandb",
                "params:model_package",
            ],
            outputs="model_package_path",
            name="build_model_package_node",
        ),
        node(
            func=log_model_package_artifact,
            inputs=["model_package_path", "params:wandb", "params:model_package"],
            outputs=None,
            name="log_model_package_artifact_node",
        ),
    ]

    return Pipeline(baseline_nodes + autogluon_nodes + comparison_nodes + package_nodes)
