# JSON Response Format Update

## Issue
The application was encountering an error when using certain LLM models (e.g., `openai/gpt-oss-120b`) with `response_format={"type": "json_object"}`:

```
Error code: 400 - {'error': {'message': "'messages' must contain the word 'json' in some form, to use 'response_format' of type 'json_object'.", 'type': 'invalid_request_error'}}
```

This error occurs because the OpenAI API requires prompts to explicitly mention "JSON" when using JSON response format.

## Solution

### 1. Created JSON Response Wrapper Function

Added a new `llm_call_with_json()` function that:
- Automatically appends JSON instructions to all prompts
- Enforces a structured response format with `copy` and `reasoning` fields
- Parses the JSON response and extracts the desired content
- Handles fallback gracefully if JSON parsing fails

```python
def llm_call_with_json(prompt: str, context: str, model: str, extract_key: str = 'copy') -> str:
    """
    Wrapper for LLM calls that handles JSON response format.
    
    Args:
        prompt: The user prompt
        context: The system context
        model: The model to use
        extract_key: The JSON key to extract from response (default: 'copy')
    
    Returns:
        Extracted text from the JSON response
    """
    json_prompt = f"""{prompt}
    
    IMPORTANT: Return your response as a JSON object with the following structure:
    {{
        "copy": "your main response text here",
        "reasoning": "brief explanation of your analysis and approach"
    }}
    """
    
    try:
        response = roains(json_prompt, context, model=model)
        
        # Try to parse as JSON
        import json
        try:
            response_json = json.loads(response)
            return response_json.get(extract_key, response_json.get('copy', response))
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw response
            print(f"Warning: Could not parse JSON response, returning raw text")
            return response
    except Exception as e:
        print(f"Error in llm_call_with_json: {e}")
        raise
```

### 2. Updated All LLM Calls

Replaced all direct `roains()` calls with `llm_call_with_json()` across the codebase:

#### ETMDataExtractor
- `extract()` method now uses `llm_call_with_json()`

#### InternetSearchService
- `search()` method now uses `llm_call_with_json()`

#### TextRefinementService
- `refine_and_proofread()` now uses `llm_call_with_json()`
- `feedback_sense_check()` now uses `llm_call_with_json()`

#### DataAnalysisService
- `process_plexos_data()` now uses `llm_call_with_json()`

#### ReportDrafter
- `add_data_to_report()` now uses `llm_call_with_json()` for both main draft and summary

#### AnalysisOrchestrator
- `analyse_default_data()` now uses `llm_call_with_json()`
- `analyse_additional_data()` now uses `llm_call_with_json()` for LLM calls and synthesis

## Response Structure

All LLM responses now follow this structure:

```json
{
    "copy": "The main response text that will be used in the report",
    "reasoning": "Brief explanation of the analysis approach and key considerations"
}
```

### Benefits

1. **API Compliance**: Ensures all prompts meet API requirements for JSON response format
2. **Structured Responses**: Provides consistent response structure across all LLM calls
3. **Traceability**: The `reasoning` field provides insight into the LLM's analysis
4. **Error Handling**: Graceful fallback if JSON parsing fails
5. **Centralized Logic**: Single point of modification for JSON handling
6. **Backward Compatible**: Extracts only the `copy` field, maintaining existing behavior

## Usage Example

```python
# Before
result = roains(prompt, context, model=self.config.base_model)

# After
result = llm_call_with_json(prompt, context, model=self.config.base_model)
```

The `llm_call_with_json()` function automatically:
1. Adds JSON structure instructions to the prompt
2. Calls the LLM
3. Parses the JSON response
4. Extracts and returns the `copy` field

## Testing Recommendations

1. Test with various model types (GPT, Gemini, Groq, etc.)
2. Verify JSON parsing handles edge cases
3. Confirm fallback works when JSON is malformed
4. Check that all report sections maintain quality
5. Validate that the `reasoning` field provides useful insights

## Future Enhancements

1. **Use Reasoning Field**: Optionally log or display the reasoning for debugging
2. **Custom Extract Keys**: Allow different keys for different use cases
3. **Response Validation**: Add schema validation for JSON responses
4. **Retry Logic**: Add retry mechanism if JSON parsing fails
5. **Caching**: Cache reasoning alongside copy for analysis

## Migration Notes

- **No Breaking Changes**: All existing code continues to work
- **Automatic JSON Handling**: Developers don't need to manually format prompts
- **Consistent Output**: All LLM calls now return clean text extracted from JSON
- **Error Messages**: Improved error messages help identify issues quickly

## Files Modified

- `src/LOLA/joule_prompt_sheet_v4.py`: Added wrapper function and updated all LLM calls

## Related Issues

This fix resolves the `400` error when using models that enforce JSON prompt requirements, particularly with the Groq API and certain OpenAI models that use `response_format={"type": "json_object"}`.
