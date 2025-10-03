# -*- coding: utf-8 -*-
"""
Created on Fri May  3 00:38:54 2024

@author: ENTSOE
"""



import openai
import pandas as pd
import sys
import yaml
from openai import OpenAI
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ai.llm_calls.get_api_keys import get_api_key

sys.path.append('utils')
# Load OpenAI API key
API_KEY = get_api_key('openai')
if API_KEY:
    os.environ['OPENAI_API_KEY'] = API_KEY
else:
    print("Failed to load OpenAI API key.")

default_ai_models_file = 'config\default_ai_models.yaml'
with open(default_ai_models_file, 'r') as f:
    ai_models_config = yaml.safe_load(f)
base_model = ai_models_config.get("open_api_assistants", "gpt-4o")

assistant = openai.beta.assistants.create(
                                        name="CSV Patcher",
                                        instructions=("You are a data-wrangling assistant. When asked, write Python with pandas and save modified files."),
                                        model=base_model,
                                        tools=[{"type": "code_interpreter"}],
                                        )
ASSISTANT_ID = assistant.id

def modify_data_file(plexos_file_path: str, data_file_name : str, user_input : str, output_file_name: str, test_mode = False) -> bytes:

    if test_mode:
        print("Running in test mode. No actual file modification will occur.")
        return {
            "status": "success",
            "file_path": 'H2\\Gas Pipeline\\Capacities_H2_Year_FID+PCI+PMI_Unlimited.csv',
        }
    """
    1) upload the CSV
    2) start a thread + send user message with code_interpreter attachment
    3) run & poll until complete
    4) download the first file the assistant returns
    """
    # 1) upload
    local_csv_path = os.path.join(plexos_file_path, data_file_name)
    output_file_path = os.path.join(plexos_file_path, output_file_name)
    
    try:
        up = openai.files.create(file=open(local_csv_path, "rb"), purpose="assistants",)

        # 2) create thread & user message
        th = openai.beta.threads.create()
        openai.beta.threads.messages.create(
                                                thread_id=th.id,
                                                role="user",
                                                content=user_input,
                                                attachments=[{"file_id": up.id, "tools": [{"type": "code_interpreter"}]}],
                                            )
                                            

        # 3) trigger run and poll
        run = openai.beta.threads.runs.create(thread_id=th.id, assistant_id=ASSISTANT_ID)
        while run.status in ("queued", "in_progress"):
            print(f"…waiting ({run.status})", end="\r")
            run = openai.beta.threads.runs.retrieve(thread_id=th.id, run_id=run.id)
        print("Run finished with status:", run.status)

        # 4) fetch assistant's reply and download the first file
        msgs = openai.beta.threads.messages.list(thread_id=th.id).data
        assistant_msg = msgs[0]  # newest first
        file_ids = [a.file_id for a in assistant_msg.attachments if hasattr(a, "file_id")]
        if not file_ids:
            raise RuntimeError("No file returned by assistant.")
        
        file_content = openai.files.content(file_ids[0])

        with open(output_file_path, "wb") as f:
            f.write(file_content.read())

        return {
            "status": "success",
            "file_path": output_file_name,
        }

    except Exception as e:
        print(f"Error during file modification: {e}")
        return f"failure: {e}"


if __name__ == "__main__":
    plexos_file_path = "C:\\Users\\ENTSOE\\Tera-joule\\Terajoule - Terajoule\\Projects\\ENTSOG\\DHEM\\National Trends\\2030\\H2\\Gas Pipeline"
    data_file_name = "Capacities_H2_FIDPCIPMIADVANCED.csv"
    user_input = "Replace every non-zero number in the value column with 999 and return the new CSV."
    output_file_name = "Capacities_H2_FIDPCIPMIADVANCED_UNLIMITED.csv"
    patched = modify_data_file(plexos_file_path, data_file_name, user_input, output_file_name)



