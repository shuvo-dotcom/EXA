import json
from datetime import datetime
from typing import List, Dict


def create_dag(tasks: List[Dict], plexos_model_location: str, plexos_model_location2: str, copy_db: bool) -> Dict:
    dag = {
        "description": "Auto-generated DAG",
        "date_created": datetime.utcnow().strftime("%Y-%m-%d"),
        "time_created": datetime.utcnow().strftime("H:%M:%S"),
        "plexos_model_location": plexos_model_location,
        "plexos_model_location2": plexos_model_location2,
        "tasks": tasks
    }
    return dag


def save_dag(dag: Dict, output_path: str):
    with open(output_path, 'w') as f:
        json.dump(dag, f, indent=4)
