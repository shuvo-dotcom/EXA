from google import genai
from google.genai import types
import os, sys, re

# Using new Gemini API (google-genai package)
client = genai.Client(api_key="AIzaSyAqUkgm5svLOizf7b-keUIfIuMTT0jsgZA")

def gemini_call(prompt, context, temperature=0.7, top_p=0.9, model = "gemini-2.5-flash"):
    gemini_prompt = f"Context: {context}\n\nUser Prompt: {prompt}"
    response = client.models.generate_content(
                                                model = model,
                                                contents = gemini_prompt,
                                                )
    response_raw = response.text
    result_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", response_raw.strip(), flags=re.IGNORECASE)
    return result_clean

def google_search(query, model  = "gemini-2.5-flash"):
    response = client.models.generate_content(
        model=model,
        contents=query,
        config= types.GenerateContentConfig(
            tools=[types.Tool(
            google_search=types.GoogleSearchRetrieval()
            )]
        )
        )
    return response.text

if __name__ == '__main__':  
    context = "You are a helpful assistant."
    prompt = "What is Trumps latest tarrifs for india?"


    response = google_search(prompt, model = "gemini-2.5-flash")
    # response = google_search(prompt)
    print(response)

