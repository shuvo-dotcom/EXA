from  open_ai_calls import run_open_ai_ns as roains


default_model = "gpt-5-mini"
test_model = "gpt-5-nano"
pro_model = "gpt-5-pro"

def run_llm_chain(user_input, context, model = default_model):
    prompt = f"""
                You are an AI assistant who is in charge of handling user queries.
                Here are the details on the request:
                    User Input: {user_input}
                    Context: {context}
                    Model: {model}

                please return you response as a json in the format most relevant to the user query.
                {{"response": <"str" | "list" | "dict" | "int" | "float">},
                    "reasoning": <"str">}
                """
    response = roains(prompt)
    context = {**context, **response}
    return context