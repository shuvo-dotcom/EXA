import sys
import os
import pandas as pd 
import json
import re
from typing import Dict, Any, List, Optional

# Add import for os and ensure project root is in sys.path for correct module resolution
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

from src.ai.llm_calls.open_ai_calls import run_open_ai_ns as roains

base_model = 'gpt-oss-120b'

def create_judgement(user_input, context, current_response = None):
    prompt = f"""
                    You are an evaluation agent. Your purpose is to evaluate the user input and context, and provide a judgement based on the response given.
                    User Input: {user_input}
                    Context: {context}
                    Current Judgement: {current_response}
                    Please return your judgement on whether this stage of the users request has been reasonably fulfilled.
                    Return as a json.
                    {{"judgement": <"fulfilled" or "not_fulfilled">, 
                        "reasoning": <your reasoning>}}
            """

    response = roains(prompt, context, model = base_model)
    return response