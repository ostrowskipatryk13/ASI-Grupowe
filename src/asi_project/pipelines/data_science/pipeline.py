from kedro.pipeline import Pipeline, node

from .nodes import evaluate_model, split_data, train_model


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
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
        ]
    )
