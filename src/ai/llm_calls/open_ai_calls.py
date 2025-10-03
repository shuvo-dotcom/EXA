# -*- coding: utf-8 -*-
"""
Created on Fri May  3 00:38:54 2024

@author: ENTSOE
"""
from pathlib import Path
import openai
import pandas as pd
import threading
import os, sys
import base64
from groq import Groq
from pydantic import BaseModel
import yaml

# Add import for os and ensure project root is in sys.path for correct module resolution
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

from src.ai.llm_calls.gemini import gemini_call
from src.ai.llm_calls.get_api_keys import get_api_key
from openai import OpenAI
import re

sys.path.append('utils')
# Load OpenAI API key
API_KEY = get_api_key('openai')
if API_KEY:
    os.environ['OPENAI_API_KEY'] = API_KEY
else:
    print("Failed to load OpenAI API key.")
# Load GROQ API key
GROQ_API_KEY = get_api_key('groq')
if GROQ_API_KEY:
    os.environ['GROQ_API_KEY'] = GROQ_API_KEY
else:
    print("Failed to load GROQ API key.")

default_ai_models_file = r'config\default_ai_models.yaml'
with open(default_ai_models_file, 'r') as f:
    ai_models_config = yaml.safe_load(f)
base_model = ai_models_config.get("base_model", "gpt-5-mini")
pro_model = ai_models_config.get("pro_model", "gpt-5")

perplexity_key = ''
# Point to the local server

import time
import simpleaudio as sa

# import requests
# import base64

# from play_chime import play_chime

# from tts import tts
# from stt import stt

# Set the API_KEY variable at the module level

#Taken from run_open_ai
def play_sound_on_off(sound_path, duration_on=1, duration_off=1, repeat=5,  play_flag=0):
    wave_obj = sa.WaveObject.from_wave_file(sound_path)
    while play_flag[0]:
        play_obj = wave_obj.play()# Play sound
        time.sleep(duration_on) # Wait for the duration the sound should be on
        play_obj.stop() # Stop the sound (if it's still playing)
        time.sleep(duration_off)  # Wait for the off duration

def play_sound_on_off(sound_path):
    if sound_path:
        wave_obj = sa.WaveObject.from_wave_file(sound_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()

def run_open_ai(message, context, temperature = 0.8, top_p = 1.0):
    global sound_playing
    chat_log = []
    sound_path = play_chime('speech_dis')# play_chime returns a valid path or None
    sound_thread = threading.Thread(target=lambda: None)# Initialize thread with a dummy function to handle cases where sound_path is None
    if sound_path:
        sound_thread = threading.Thread(target=play_sound_on_off, args=(sound_path,))
        sound_thread.start()
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content":context},
            {"role": "user", "content": message}, ],
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,)
    AI_response = response.choices[0].message.content
    chat_log.append({'role': 'assistant', 'content': AI_response.strip('\n').strip()})
    if sound_thread.is_alive():
        sound_thread.join()
    return AI_response

def open_ai_search(message, model = "gpt-4o-mini-search-preview"):
    completion = openai.chat.completions.create(
    model="gpt-4o-search-preview",
    messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
    )
    return completion.choices[0].message.content

def open_ai_base_call(message, chat_log, model = "gpt-4o-mini"):
    chat_log = []
    response = openai.chat.completions.create(
        model=model,       
        messages=history,  # Use the full history
        top_p=top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,)
    AI_response = response.choices[0].message.content
    chat_log.append({'role': 'assistant', 'content': AI_response.strip('\n').strip()})
    return AI_response

def _call_groq_tts(message, file_name):
    """Handle PlayAI TTS via Groq."""
    client = Groq(api_key=GROQ_API_KEY)
    speech_file_path = file_name
    response = client.audio.speech.create(
        model="playai-tts",
        voice="Fritz-PlayAI",
        response_format="wav",
        input=message,
    )
    response.stream_to_file(speech_file_path)
    os.system(f"start {speech_file_path}")
    return speech_file_path


def _call_groq_chat(model, context, message):
    """Handle Groq chat completions."""
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f'{context}. {message}'}],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )
    response = completion.choices[0].message.content
    print(response or "", end="")
    return response


def _call_gpt(model, history, top_p):
    """Handle standard GPT model calls."""
    response = openai.chat.completions.create(
        model=model,
        messages=history,
        top_p=top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response.choices[0].message.content


def _call_o1_o3(model, context, message):
    """Handle O1 and O3 reasoning models."""
    # Remove newline characters for these models
    clean_message = message.replace('\n', '')
    clean_context = context.replace('\n', '')
    
    response = openai.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f'{clean_context}. {clean_message}'
        }]
    )
    return response.choices[0].message.content


def _extract_json_from_response(response_text):
    """Extract JSON from various response formats."""
    try:
        # 1. Look for markdown ```json ... ``` block
        json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # 2. Look for json.loads('...') constructs
        loads_match = re.search(r"""json\.loads\(\s*['"](\{[\s\S]*?\})['"]s*\)""", response_text, re.DOTALL)
        if loads_match:
            return loads_match.group(1).strip().replace('\\"', '"')
        
        # 3. Handle escaped JSON string
        if response_text.startswith('"') and response_text.endswith('"'):
            return response_text[1:-1].replace('\\"', '"').replace('\\\\', '\\')
        
        # 4. Fallback: Find JSON object by curly braces
        curly_match = re.search(r'\{[\s\S]*\}', response_text)
        if curly_match:
            return curly_match.group(0).strip()
        
        return response_text
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
        return response_text


def _call_lm_studio(model, history, temperature, top_p):
    """Handle LM Studio local model calls."""
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    completion = client.chat.completions.create(
        model=model,
        messages=history,
        temperature=temperature,
        top_p=top_p,
    )
    response = completion.choices[0].message.content.strip()
    return _extract_json_from_response(response)


def _call_deepseek(model, context, message):
    """Handle DeepSeek model calls."""
    client = OpenAI(api_key="sk-bba69b9e3f4c40529602d89878f7b6fa", base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": message},
        ],
        stream=False
    )
    return response.choices[0].message.content


def _call_perplexity(context, message):
    """Handle Perplexity Sonar model calls."""
    client = OpenAI(api_key=perplexity_key, base_url="https://api.perplexity.ai")
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model="sonar-pro",
        messages=messages,
    )
    return response.choices[0].message.content


def _handle_fallback(model, message, context, temperature, top_p):
    """Handle fallback when primary model fails."""
    if 'gemini' in model:
        fallback_model = "gpt-5" if 'pro' in model else "gpt-5-mini"
        response = openai.chat.completions.create(
            model=fallback_model,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": message},
            ],
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content
    
    elif model in ['o1', 'o3', 'o4', 'gpt-5']:
        return gemini_call(message, context, model="gemini-2.5-pro")
    
    else:
        return gemini_call(message, context, temperature=temperature, top_p=top_p, model="gemini-2.5-flash")


def run_open_ai_ns(message, context, temperature = 0.3, top_p = 1.0, model = base_model, history=None, reasoning_effort="medium", file_name = Path(__file__).parent / "speech.wav"):
    """
    Universal LLM caller supporting multiple model providers.
    
    Args:
        message: User prompt
        context: System context/instructions
        temperature: Sampling temperature (0.0-1.0)
        top_p: Nucleus sampling parameter
        model: Model identifier (e.g., 'gpt-4', 'gemini-pro', 'groq-*')
        history: Conversation history (list of message dicts)
        reasoning_effort: Reasoning effort for O1/O3 models
        file_name: Output file path for TTS models
    
    Returns:
        Model response (string or file path for TTS)
    """
    if history is None:
        history = []
    
    # Build conversation history
    history.append({"role": "system", "content": context})
    history.append({"role": "user", "content": message})
    
    try:
        # Route to appropriate model provider
        if model == 'playai-tts':
            return _call_groq_tts(message, file_name)
        
        elif 'groq' in model or 'oss' in model or 'kimi' in model:
            return _call_groq_chat(model, context, message)
        
        elif 'gpt' in model:
            return _call_gpt(model, history, top_p)
        
        elif 'o1' in model or 'o3' in model:
            return _call_o1_o3(model, context, message)
        
        elif 'studio' in model:
            return _call_lm_studio(model, history, temperature, top_p)
        
        elif 'deepseek' in model:
            return _call_deepseek(model, context, message)
        
        elif 'sonar' in model:
            return _call_perplexity(context, message)
        
        elif 'gemini' in model:
            return gemini_call(message, context, temperature=temperature, top_p=top_p, model=model)
        
        elif 'test' in model:
            return 'This is a test call'
        
        else:
            raise ValueError(f"Unsupported model: {model}")
    
    except Exception as e:
        print(f"Error calling {model}: {e}. Attempting fallback...")
        return _handle_fallback(model, message, context, temperature, top_p)

def run_open_ai_json(message, context, temperature = 0.7, top_p = 1.0, model = "gpt-4o-mini"):
    chat_log = []
    response = openai.chat.completions.create(
        model=model,
        response_format = {"type": "json_object"},
        
        messages=[
            {"role": "system", "content":context},
            {"role": "user", "content": message}, ],
        temperature=temperature,
        
        top_p=top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,)
    AI_response = response.choices[0].message.content
    chat_log.append({'role': 'assistant', 'content': AI_response.strip('\n').strip()})
    return AI_response

def run_open_ai_o1(message, context, model="o1-mini"):
    # Remove newline characters from context and message
    message = message.replace('\n', '')
    context = context.replace('\n', '')

    # Create the completion with the correct structure
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user", 
                "content": f'{context}. {message}'
            }
        ]
    )
    return response.choices[0].message.content

def ai_gap_filler(prompt):
    user_message = """Please make an assessment of whether this prompt will take more than 3 seconds to from the openAI API gpt-4o to give a response. 
                If so please stall while the request is being processed. The output of the call will be text-to-speech, therefore the response must be very short, very roughly in the 20 word range.\
                Sumamrize the request reassure the user you know what the request is and that it is being processed. E.g. 'Hey, i got your request for xyz. I'll get back to you in a sec' Don't include timings.
                Here is your prompt: {prompt}. 
                Return either the stall_response or no_stall_required. Remember your response will be directly converted to tts, so ensure no additional text.
                """
    chat_log = []
    chat_log.append({"role": "user", "content": user_message})
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages = chat_log    )
    AI_response = response.choices[0].message.content
    return AI_response

def open_ai_copywriter(purpose, target_audience, tone_voice, format_, length, example, position, prompt):
    print(prompt)
    response = openai.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": f"You are an experienced copywriter tasked with writing compelling content. \
                Please use the following inputs: Purpose = {purpose}, Target Audience = {target_audience}, Tone and Voice = {tone_voice}, \
                Format = {format_}, Length = {length}, Position on topic = {position}, Examples of content = {example}. \
                Ensure the content is engaging, adheres to the provided specifications, and reflects the intended tone and style. \
                Please respond in the format. Amount of Input tokens used:, Amount of Output tokens used:, Final Copy:"},
            {"role": "user", "content": prompt}],
        temperature=.5,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,)
    return response.choices[0].message.content

def load_category_descriptions(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    # Create a dictionary with categories as keys and descriptions as values
    return dict(zip(df['Key'], df['Description']))

def open_ai_categorisation(question, function_map, level = None):
    categories = load_category_descriptions(function_map)# Load category descriptions
    #if level == 'task list', remove Create task list from the categories
    if level == 'task list':
        categories.pop('Create task list')

    category_info = ", ".join([f"'{key}': {desc}" for key, desc in categories.items()])# Prepare the message part with category descriptions for better understanding
    response = openai.chat.completions.create(
        model="o3-mini",
        messages=[
            {"role": "system", "content": f"""You are an assistant trained to categorize questions into specific functions. \
              Here are the categories with descriptions: {category_info}. \
              If none of the categories are appropriate, categorize as 'Uncategorized'. Please respond with only the category from the list given, no additional text.
              ensure names are taken from the list provided. do not add additiona punctuation or text e.g. do not end with full stops
              """},
            {"role": "user", "content": question} ])
    category = response.choices[0].message.content
    return category

def ai_chat_session(prompt = None):
    # Point to the local server
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    history = [
        {"role": "system", "content": "You are an friendly, intelligent assistant. You always provide well-reasoned answers that are both correct and helpful. Keep your responses concise and to the point."},
        {"role": "user", "content": f"Hello, you have been requested. Here is the prompt: {prompt}"},
    ]

    while True:
        completion = client.chat.completions.create(
            model="bartowski/Phi-3-medium-128k-instruct-GGUF",
            messages=history,
            temperature=0.7,
            stream=True,
        )

        new_message = {"role": "assistant", "content": ""}
        
        for chunk in completion:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                new_message["content"] += chunk.choices[0].delta.content

        history.append(new_message)
        
        # Uncomment to see chat history
        # import json
        # gray_color = "\033[90m"
        # reset_color = "\033[0m"
        # print(f"{gray_color}\n{'-'*20} History dump {'-'*20}\n")
        # print(json.dumps(history, indent=2))
        # print(f"\n{'-'*55}\n{reset_color}")

        print()
        history.append({"role": "user", "content": input("> ")})

def ai_spoken_chat_session(prompt = None):
    # Point to the local server
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    history = [
        {"role": "system", "content": """You are an friendly, intelligent assistant. You always provide well-reasoned answers that are both correct and helpful. 
                                         I just called you and asked you to have a chat. Respond to let me know you have heard me and we will start                                 
                                        Keep your responses concise and to the point. If there is no prompt given just respond simply"""},
        {"role": "user", "content": "{prompt}"},
    ]

    while True:
        completion = client.chat.completions.create(
            model="bartowski/Phi-3-medium-128k-instruct-GGUF",
            messages=history,
            temperature=0.7,
            stream=True,
        )

        new_message = {"role": "assistant", "content": ""}
        
        for chunk in completion:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                new_message["content"] += chunk.choices[0].delta.content

        tts(new_message["content"])
        history.append(new_message)
        
        # Uncomment to see chat history
        # import json
        # gray_color = "\033[90m"
        # reset_color = "\033[0m"
        # print(f"{gray_color}\n{'-'*20} History dump {'-'*20}\n")
        # print(json.dumps(history, indent=2))
        # print(f"\n{'-'*55}\n{reset_color}")

        print()
        my_prompt = stt()
        history.append({"role": "user", "content": my_prompt})

        # Check if the user's phrase indicates the end of the conversation
        if my_prompt.lower() in ["bye", "goodbye", "end", "stop", "see you later"]:
            break

def openai_vision(image_path, context ):
    # Load the image
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Call the OpenAI Vision API
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
                }

    payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": context
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                        }
                    ]
                    }
                ],
                }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return (response.json()['choices'][0]['message']['content'])

# Function to continue the conversation with updated context
def continue_conversation(user_input, context, instruction, model = "gpt-4o-mini"):
    # Combine logs for context
    # context = "\n".join(log)
    
    # Call GPT-4 with context-aware prompt
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": context}
        ],
        stream=True  # Enable streaming if available
    )

    nova_response = ""
    # Collect and return the response
    for chunk in response:
        # Assuming `chunk` is an instance of `ChatCompletionChunk` and has attributes `choices` and `delta`
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="")
            nova_response += content
    
    return nova_response

if __name__ == '__main__':
    # import os
    # print("HTTP_PROXY:", os.environ.get("HTTP_PROXY"))
    # print("HTTPS_PROXY:", os.environ.get("HTTPS_PROXY"))
    # prompt = 'Give 10 ways to loose weight, with a workout regime and meal plan, considering my weight is 100 kg and height is 180 meters and i am 26'
    # image_path = r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Audio & Graphics\Graphics\AI Assistants\Nova.jpeg'
    # openai_vision(image_path)
    context = "You are a helpful assistant."
    prompt = "what is the nuclear capacity in the UK?"
    response = run_open_ai_ns(prompt, context, model = "playai-tts")
    print(response)
    # response = openai_cot(prompt, context)
    # print(response.final_answer)
    # open_ai_search(prompt)
    # print(run_open_ai_ns(prompt, context, model = "gpt-4o-mini-search-preview-2025-03-11", max_tokens = 500))

