# -*- coding: utf-8 -*-
"""
Created on Fri May  3 00:38:54 2024

@author: ENTSOE
"""
import openai
import pandas as pd
import os, sys

top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

from src.ai.llm_calls.get_api_keys import get_api_key
from openai import OpenAI

sys.path.append('utils')
# Load OpenAI API key
API_KEY = get_api_key('openai')
if API_KEY:
    os.environ['OPENAI_API_KEY'] = API_KEY
else:
    print("Failed to load OpenAI API key.")

assistant = openai.beta.assistants.create(
                                        name="CSV Patcher",
                                        instructions=("You are a data-wrangling assistant. When asked, write Python with pandas and save modified files."),
                                        model="gpt-4o",
                                        tools=[{"type": "code_interpreter"}],
                                        )
ASSISTANT_ID = assistant.id

def modify_data_file(local_csv_path: str, user_input : str) -> bytes:
    """
    1) upload the CSV
    2) start a thread + send user message with code_interpreter attachment
    3) run & poll until complete
    4) download the first file the assistant returns
    """
    # 1) upload
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
    return openai.files.content(file_ids[0])


if __name__ == "__main__":
    local_csv = "C:\\Users\\ENTSOE\\Tera-joule\\Terajoule - Terajoule\\Projects\\ENTSOG\\DHEM\\NT2040\\H2\\Gas Pipeline\\Capacities_H2_Year_FID+PCI+PMI.csv"
    user_input = "Replace every non-zero number in the value column with 999 and return the new CSV."
    patched = modify_data_file(local_csv, user_input, )

    out_path = "C:\\Users\\ENTSOE\\Tera-joule\\Terajoule - Terajoule\\Projects\\ENTSOG\\DHEM\\NT2040\\H2\\Gas Pipeline\\Capacities_H2_Year_FID+PCI+PMI_UNLIMITED.csv"
    with open(out_path, "wb") as f:
        f.write(patched.read())
