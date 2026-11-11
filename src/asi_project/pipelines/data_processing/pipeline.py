from kedro.pipeline import Pipeline, node

from .nodes import filter_data, normalize_data, validate_data


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=validate_data,
                inputs="raw_data",
                outputs="validated_data",
                name="validate_data_node",
            ),
            node(
                func=normalize_data,
                inputs="validated_data",
                outputs="normalized_data",
                name="normalize_data_node",
            ),
            node(
                func=filter_data,
                inputs="normalized_data",
                outputs="prepared_data",
                name="filter_data_node",
            ),
        ]
    )
