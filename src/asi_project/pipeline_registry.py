from kedro.pipeline import Pipeline

from asi_project.pipelines.data_processing import create_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    data_processing = create_pipeline()

    return {
        "__default__": data_processing,
        "data_processing": data_processing,
    }
