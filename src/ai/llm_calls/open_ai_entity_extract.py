from pydantic import BaseModel
from openai import OpenAI
import sys, os
from groq import Groq  # import Groq client for GROQ model
import json
from textwrap import dedent
import openai
from enum import Enum

sys.path.append('utils')
from src.ai.llm_calls.get_api_keys import get_api_key

# Load OpenAI API key
API_KEY = get_api_key('openai')
if API_KEY:
    os.environ['OPENAI_API_KEY'] = API_KEY
else:
    print("Failed to load OpenAI API key.")
client = OpenAI()
# Load GROQ API key for Groq client
GROQ_API_KEY = get_api_key('groq')
if GROQ_API_KEY:
    os.environ['GROQ_API_KEY'] = GROQ_API_KEY
else:
    print("Failed to load GROQ API key.")
# thread = client.beta.threads.create()

class extract_variable(BaseModel):
    variable: str

def get_single_response(prompt, context, user_input, value_list, MODEL = "o3-mini"):
    value_list_array = list(value_list)  # Convert set to list

    product_search_function = {
        "type": "function",
        "function": {
            "name": "value_search",
            "description": "Find the closest matches from the options.",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "string",
                        "enum": value_list_array  # Use the list here
                    }
                },
                "additionalProperties": False,
                "required": ["values"]
            }
        },
    "strict": True
    }

    if 'gpt' in MODEL or 'o1' in MODEL or 'o3' in MODEL:
        response = client.chat.completions.create(
            model = MODEL,
            messages=[
                {
                    "role": "system",
                    "content": context
                },
                {
                    "role": "user",
                    "content": f"Prompt: {prompt}\n USER INPUT: {user_input}"
                }
            ],
            tools = [product_search_function]
        )

    if 'deepseek' in MODEL:
        deepseek = OpenAI(api_key="sk-bba69b9e3f4c40529602d89878f7b6fa", base_url="https://api.deepseek.com")
        response = deepseek.chat.completions.create(
            model = MODEL,
            messages=[
                {
                    "role": "system",
                    "content": context
                },
                {
                    "role": "user",
                    "content": f"Prompt: {prompt}\n USER INPUT: {user_input}"
                }
            ],
            tools = [product_search_function]
        )
    
    if 'qwen' in MODEL:
        # Use Groq client with environment-loaded API key
        groq_client = Groq()
        response = groq_client.chat.completions.create(
            model = MODEL,
            messages=[
                {
                    "role": "system",
                    "content": context
                },
                {
                    "role": "user",
                    "content": f"Prompt: {prompt}\n USER INPUT: {user_input}"
                }
            ],
            tools = [product_search_function]
        )

    output = response.choices[0].message.tool_calls
    return output

def get_single_response_pydantic(prompt, context, user_input, value_list, MODEL = "gpt-4o-mini"):
    value_list_array = list(value_list)  # Convert set to list

    class Category(str, Enum):
        yes = "yes"
        no = "no"

    class ProductSearchParameters(BaseModel):
        category: Category

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": dedent(prompt)
            },
            {
                "role": "user",
                "content": f"CONTEXT: {context}\n USER INPUT: {user_input}"
            }
        ],
        tools=[
            openai.pydantic_function_tool(ProductSearchParameters, name="product_search", description="Search for a match in the database")
        ]
    )

    return response.choices[0].message.tool_calls

def get_multi_response(prompt, context, user_input, value_list, MODEL = "gpt-4o-mini"):
    if 'deepseek' in MODEL:
        client = OpenAI(api_key="sk-bba69b9e3f4c40529602d89878f7b6fa", base_url="https://api.deepseek.com")

    value_list_array = list(value_list)  # Convert set to list

    product_search_function = {
        "type": "function",
        "strict": True,
        "function": {
            "name": "value_search",
            "description": "Search for a match in the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "enum": value_list_array  # Use the list here

                    }
                },
                "required": ["values"]
            }
        }
    }

    response = client.chat.completions.create(
        model = MODEL,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": f"CONTEXT: {context}\n USER INPUT: {user_input}"
            }
        ],
        tools=[product_search_function]
    )
    output = response.choices[0].message.tool_calls
    return output

class timeslice_structure(BaseModel):
    start_id: str
    end_id: str

def get_timeslice(prompt, context, MODEL = "gpt-4o-mini"):
    completion = client.beta.chat.completions.parse(
                                                    model=MODEL,
                                                    messages=[
                                                        {"role": "system", "content": context},
                                                        {"role": "user", "content": prompt},
                                                    ],
                                                    response_format=timeslice_structure,
                                                    )

    event = completion.choices[0].message.parsed
    return event

def get_closest_match(prompt, context, value_list = None, MODEL = "gpt-4.1-mini", user_input = None, type = 'Normal', response_format = 'single'):
    if type == 'Normal':
        count = 0 
        while True:
            if count == 3:
                return "Error: Failed to get response from the model."
            try:
                if response_format == 'single':
                    result = get_single_response(prompt, context, user_input, value_list, MODEL)
                else:
                    result = get_multi_response(prompt, context, user_input, value_list, MODEL)
                pass 
            
                output = print_tool_call(result)
                return output
            except Exception as e:
                count += 1
                return e
                
    elif type == 'Timeslice':
        result = get_timeslice(prompt, context,  MODEL)
        return result

    # result = get_response(prompt, context, MODEL)

def print_tool_call(tool_call, key = 'values'):
    outputs = []
    object_returned = len(tool_call) 
    if object_returned > 1:
        for x in range(0,(len(tool_call))):
            args = tool_call[x].function.arguments
            outputs.append(json.loads(args)[key])
        return outputs
    else:
        args = tool_call[0].function.arguments
        
        return json.loads(args)[key]

def get_dictionary_tree_response(user_input, context, objectives, objective_tasks, MODEL = "gpt-4o-mini"):
    product_search_prompt = f'''
    You are a keyword similarity expert, specialized in finding the perfect match for a user.
    You will be provided with a user input and additional context.
    You are equipped with a tool to search a database that match the user's request.
    Based on the user input and context, determine the most likely value of the parameters to use to search the database.
    
    Here are the different categories that are available on the website:
    {objective_tasks}       
    '''
    product_search_function = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for a match in the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The broad category of the product",
                        "enum": [objectives]
                    },
                    "Title": {
                        "type": "string",
                        "description": "The sub category of the objective, within the broader category",
                    },
                },
                "required": ["category", "Title"],
                "additionalProperties": False,
            }
        },
        "strict": True
    }

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": product_search_prompt
            },
            {
                "role": "user",
                
                "content": f"CONTEXT: {context}\n USER INPUT: {user_input}"
            }
        ],
        tools=[product_search_function]
    )

    #extract the tool calls from the response
    tool_calls = response.choices[0].message.tool_calls
    output = print_tool_call(tool_calls, key = 'subcategory')
    return output
