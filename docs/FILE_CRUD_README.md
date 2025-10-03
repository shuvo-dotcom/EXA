# File CRUD Operations System

A comprehensive file management system that performs Create, Read, Update, Delete operations on various file types with AI integration for intelligent content processing and analysis.

## Features

- **Multi-format Support**: Works with TXT, JSON, CSV, XLSX, XML files
- **AI Integration**: Uses OpenAI models for content generation, analysis, and intelligent updates
- **Code Interpreter**: Leverages OpenAI Code Interpreter for complex data transformations
- **Safe Operations**: Creates new files instead of overwriting existing ones
- **Error Handling**: Comprehensive error handling and user feedback
- **Flexible Input**: Accepts string, dictionary, or file-based input data

## Installation

### Prerequisites

```bash
pip install pandas openai pathlib
```

### Required Files

The system requires these AI integration modules:
- `src/ai/open_ai_calls.py` - For general AI text processing
- `src/ai/openai_code_interpretor_assistant.py` - For data file processing

## Usage

### Basic Usage

```python
from file_CRUD import FileCRUD

# Initialize the CRUD handler
crud = FileCRUD()

# Perform operations
result = crud.execute_crud_operation(
    action="create",  # or "read", "update", "delete"
    input_data="Your data here",
    output_location="path/to/output.txt",
    output_structure="txt",
    user_request="AI instruction here"
)

print(result)
```

### Parameters

#### `execute_crud_operation()` Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | str | Yes | CRUD operation: 'create', 'read', 'update', 'delete' |
| `input_data` | str/dict/None | No | Data to process |
| `input_location` | str | No | Path to input file |
| `output_location` | str | No | Path for output file |
| `output_structure` | str | No | Output format: 'txt', 'json', 'csv', 'xlsx', 'xml' |
| `output_extension` | str | No | File extension for output |
| `user_request` | str | No | AI instruction for processing |
| `ai_context` | str | No | Context for AI calls |

## Operations

### 1. CREATE Operation

Creates new files with optional AI-generated content.

#### Example 1: Simple Text File
```python
result = crud.execute_crud_operation(
    action="create",
    input_data="Hello, World!",
    output_location="output/hello.txt",
    output_structure="txt"
)
```

#### Example 2: AI-Generated Configuration
```python
result = crud.execute_crud_operation(
    action="create",
    output_location="config/app.json",
    output_structure="json",
    user_request="Create a configuration file for a web API with database settings, logging, and rate limiting",
    ai_context="You are a software configuration expert."
)
```

#### Example 3: Data File with Code Interpreter
```python
result = crud.execute_crud_operation(
    action="create",
    output_location="data/report.csv",
    output_structure="csv",
    user_request="Create a sales report with sample data for Q4 2024 including regions, products, and revenue"
)
```

### 2. READ Operation

Reads and analyzes file content with optional AI analysis.

#### Example 1: Simple File Reading
```python
result = crud.execute_crud_operation(
    action="read",
    input_location="data/employees.csv"
)
content = result['content']
```

#### Example 2: AI-Powered Analysis
```python
result = crud.execute_crud_operation(
    action="read",
    input_location="data/sales_data.csv",
    user_request="Analyze this sales data and identify top-performing regions and trending products",
    ai_context="You are a business intelligence analyst."
)
analysis = result['ai_analysis']
```

### 3. UPDATE Operation

Updates files by creating new versions with modifications.

#### Example 1: Simple Content Addition
```python
result = crud.execute_crud_operation(
    action="update",
    input_data="Additional content to append",
    input_location="documents/report.txt",
    output_location="documents/report_v2.txt",
    user_request="Add a conclusion section summarizing the key findings"
)
```

#### Example 2: Data Transformation
```python
result = crud.execute_crud_operation(
    action="update",
    input_location="data/employees.csv",
    output_location="data/employees_enhanced.csv",
    user_request="Add calculated columns for tenure in years and salary grades based on current salary",
    output_structure="csv"
)
```

### 4. DELETE Operation

Safely deletes files with explicit confirmation required.

#### Example 1: Safe Deletion
```python
result = crud.execute_crud_operation(
    action="delete",
    input_location="temp/old_file.txt",
    user_request="Please delete this temporary file permanently"
)
```

**Note**: The word "delete" must be present in the `user_request` for the operation to proceed.

## Return Format

All operations return a dictionary with the following structure:

```python
{
    "status": "success" | "error" | "warning",
    "message": "Description of the operation result",
    "content": "File content (for read operations)",
    "ai_analysis": "AI analysis result (when requested)",
    "output_path": "Path to created/updated file",
    "ai_result": "Result from code interpreter (when used)"
}
```

## Supported File Formats

| Format | Read | Write | AI Processing | Code Interpreter |
|--------|------|-------|---------------|------------------|
| TXT | ✅ | ✅ | ✅ | ❌ |
| JSON | ✅ | ✅ | ✅ | ❌ |
| CSV | ✅ | ✅ | ✅ | ✅ |
| XLSX | ✅ | ✅ | ✅ | ✅ |
| XML | ✅ | ✅ | ✅ | ❌ |

## AI Integration

### OpenAI Integration

The system uses `run_open_ai_ns()` function for:
- Content generation
- File analysis
- Update instructions
- Smart content processing

### Code Interpreter

For complex data operations, the system uses `modify_data_file()` which:
- Handles CSV and Excel files
- Performs data transformations
- Executes Python code for data processing
- Creates new files with processed data

## Error Handling

The system includes comprehensive error handling:

- File not found errors
- Invalid operation types
- AI service failures
- File permission issues
- Format conversion errors

## Safety Features

1. **No Overwriting**: Update operations create new files instead of modifying originals
2. **Delete Confirmation**: Deletion requires explicit confirmation in the request
3. **Path Validation**: Automatic directory creation for output paths
4. **Graceful Degradation**: Falls back to basic operations if AI services are unavailable

## Examples and Demonstrations

Run the demonstration script to see the system in action:

```bash
python demo_file_CRUD.py
```

Choose between:
1. **Automated Demo**: Shows all operations with sample data
2. **Interactive Demo**: Allows you to test operations interactively

## Use Cases

### 1. Document Management
- Generate reports with AI assistance
- Analyze document content
- Update documents with new information
- Archive and organize files

### 2. Data Processing
- Transform CSV/Excel files
- Generate data summaries
- Clean and validate data
- Create data visualizations

### 3. Configuration Management
- Generate configuration files
- Update settings across environments
- Validate configuration syntax
- Document configuration changes

### 4. Content Creation
- Generate documentation
- Create templates
- Process text content
- Format and structure data

## Advanced Features

### Custom AI Context

```python
custom_context = "You are a financial analyst specializing in risk assessment."

result = crud.execute_crud_operation(
    action="read",
    input_location="portfolio.csv",
    user_request="Assess the risk profile of this investment portfolio",
    ai_context=custom_context
)
```

### Batch Processing

```python
# Process multiple files
files_to_process = ["data1.csv", "data2.csv", "data3.csv"]

for file_path in files_to_process:
    result = crud.execute_crud_operation(
        action="update",
        input_location=file_path,
        output_location=f"processed/{Path(file_path).name}",
        user_request="Standardize column names and data formats"
    )
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure AI integration modules are in the correct path
2. **API Key Issues**: Verify OpenAI API keys are properly configured
3. **File Permission Errors**: Check read/write permissions for target directories
4. **Memory Issues**: Large files may require chunked processing

### Debug Mode

Enable debug output by checking result messages:

```python
result = crud.execute_crud_operation(...)
if result['status'] == 'error':
    print(f"Error: {result['message']}")
```

## Contributing

When extending the system:

1. Add new file format handlers in the `supported_formats` dictionary
2. Implement `_handle_format()` methods for new formats
3. Add comprehensive error handling
4. Update documentation and examples

## License

This system is part of the AI Architecture project and follows the project's licensing terms.
