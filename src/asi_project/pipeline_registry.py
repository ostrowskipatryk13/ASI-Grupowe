from kedro.pipeline import Pipeline

from asi_project.pipelines.data_processing import (
    create_pipeline as create_data_processing_pipeline,
)
from asi_project.pipelines.data_science import (
    create_pipeline as create_data_science_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    data_processing = create_data_processing_pipeline()
    data_science = create_data_science_pipeline()

    return {
        "__default__": data_processing + data_science,
        "data_processing": data_processing,
        "data_science": data_science,
    }
