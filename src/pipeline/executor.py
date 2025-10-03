import json
from src.pipeline.archive.pipeline_executor_v03 import PipelineExecutor

def execute_pipelines(function_registry_path: str, pipeline_file_paths: list):
    executor = PipelineExecutor(function_registry_path=function_registry_path)
    results = []
    for path in pipeline_file_paths:
        with open(path, 'r') as f:
            result = executor.loop_dags(json.load(f))
            results.append(result)
    return results
