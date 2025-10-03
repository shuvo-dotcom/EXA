# Quick Reference: Using llm_call_with_json()

## Function Signature

```python
def llm_call_with_json(prompt: str, context: str, model: str, extract_key: str = 'copy') -> str
```

## Parameters

- **prompt** (str): Your main prompt/question to the LLM
- **context** (str): System context/instructions for the LLM
- **model** (str): Model identifier (e.g., 'gpt-5-mini', 'gemini-2.5-pro')
- **extract_key** (str, optional): JSON key to extract (default: 'copy')

## Returns

- **str**: The extracted text from the JSON response

## Expected JSON Response Format

The LLM will return:
```json
{
    "copy": "Your main content here",
    "reasoning": "Explanation of the approach"
}
```

## Examples

### Basic Usage

```python
prompt = "Summarize the energy data for Germany"
context = "You are an energy analyst"
result = llm_call_with_json(prompt, context, model='gpt-5-mini')
# Returns: "Germany's energy sector shows..."
```

### Extract Different Field

```python
# Get the reasoning instead of copy
reasoning = llm_call_with_json(
    prompt="Analyze this data",
    context="You are an analyst",
    model='gpt-5',
    extract_key='reasoning'
)
```

### Full Example from Report Generation

```python
prompt = f"""Please give a concise summary of the data extracted. 
            country: {country}. 
            sector: {sector}.
            data: {json_data_output}
            Compare the data between 2019 and 2050. 
            """
            
context = f"""{self.config.main_context}. 
            You are currently drafting chapter {sub_task_id}.
            Instruction: {instruction}.
            """

etm_results = llm_call_with_json(prompt, context, model=self.config.base_model)
```

## Error Handling

The function handles errors gracefully:

```python
try:
    result = llm_call_with_json(prompt, context, model='gpt-5')
except Exception as e:
    print(f"Error: {e}")
    # Handle error appropriately
```

### Fallback Behavior

If JSON parsing fails, the function:
1. Prints a warning message
2. Returns the raw response text
3. Continues execution without crashing

## Common Use Cases

### 1. Data Analysis

```python
result = llm_call_with_json(
    prompt="Analyze this PLEXOS data and identify key trends",
    context="You are a power systems analyst",
    model=self.config.sota_model
)
```

### 2. Text Refinement

```python
refined = llm_call_with_json(
    prompt=f"Refine this text: {draft_text}",
    context="You are a professional editor",
    model=self.config.writer_model
)
```

### 3. Internet Search Synthesis

```python
synthesis = llm_call_with_json(
    prompt=f"Synthesize these search results: {results}",
    context="You are a research analyst",
    model=self.config.base_model
)
```

### 4. Report Summary

```python
summary = llm_call_with_json(
    prompt=f"Summarize this section: {section_text}",
    context="Create a concise summary",
    model=self.config.base_model
)
```

## Migration from roains()

### Before
```python
result = roains(prompt, context, model=model)
```

### After
```python
result = llm_call_with_json(prompt, context, model=model)
```

That's it! The function automatically handles JSON formatting.

## Benefits at a Glance

✅ **API Compliant**: Meets JSON response format requirements  
✅ **Consistent**: All responses follow same structure  
✅ **Traceable**: Reasoning field provides transparency  
✅ **Robust**: Handles parsing errors gracefully  
✅ **Simple**: Drop-in replacement for roains()  

## Troubleshooting

### Issue: Getting raw text instead of clean output
**Solution**: Check if the LLM is actually returning JSON. The function will fall back to raw text if JSON parsing fails.

### Issue: Need both copy and reasoning
**Solution**: Call the function twice with different `extract_key` values, or modify to return both fields.

```python
copy = llm_call_with_json(prompt, context, model, extract_key='copy')
reasoning = llm_call_with_json(prompt, context, model, extract_key='reasoning')
```

### Issue: Model not supporting JSON format
**Solution**: The function will still work - it just returns the raw response if JSON parsing fails.

## Notes

- The function automatically adds JSON instructions to your prompt
- You don't need to modify your prompts manually
- The original `roains()` function is still called internally
- Works with all model types (GPT, Gemini, Groq, DeepSeek, etc.)

## Related Documentation

- See `JSON_RESPONSE_FORMAT_UPDATE.md` for detailed implementation
- See `REFACTORING_SUMMARY_open_ai_calls.md` for LLM call improvements
