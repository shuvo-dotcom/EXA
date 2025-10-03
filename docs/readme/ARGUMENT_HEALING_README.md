# Pipeline Executor V05 - Argument Healing System

## Overview

The Pipeline Executor V05 introduces an advanced **Argument Healing** system that automatically detects and resolves argument issues when pipeline tasks fail. This system uses multiple strategies to heal broken arguments, including DAG context analysis and LLM-based resolution.

## Key Features

### 🏥 Argument Healing
- **Automatic Detection**: Identifies when function calls fail due to argument issues
- **Context Analysis**: Searches previous DAG task outputs for missing values
- **LLM Resolution**: Uses available functions and AI to resolve complex argument problems
- **Multi-Attempt Recovery**: Tries multiple healing strategies with configurable retry attempts
- **History Tracking**: Maintains detailed logs of all healing attempts for debugging

### 🔄 Healing Strategies

#### 1. DAG Context Healing
- Searches task outputs for missing argument values
- Resolves references like `task_outputs.previous_task.value`
- Matches argument names using fuzzy logic and common patterns
- Handles nested data structures and complex references

#### 2. LLM-Based Healing
- Analyzes failed arguments and suggests resolution strategies
- Uses function registry to identify available data sources
- Can trigger additional function calls to gather missing information
- Provides intelligent argument mapping and type conversion

#### 3. Function Registry Integration
- Leverages available functions for data retrieval
- Supports file search, internet search, RAG search capabilities
- Automatically suggests appropriate functions based on argument requirements
- Executes healing functions and integrates results into argument resolution

## Architecture

### ArgumentHealer Class
```python
class ArgumentHealer:
    def heal_arguments(self, user_input, task_outputs, current_inputs, 
                      original_inputs, task_def, task_id, max_attempts=2)
```

**Core Methods:**
- `_heal_from_dag_context()`: Analyzes DAG execution context
- `_heal_with_llm_and_functions()`: Uses LLM for intelligent resolution
- `_search_task_outputs_for_value()`: Fuzzy matching for argument values
- `_parse_and_execute_llm_response()`: Executes LLM-suggested actions

### PipelineExecutorV05 Class
```python
class PipelineExecutorV05:
    def run_dag_with_healing(self, dag_file, user_input="", initial_context=None)
    def execute_task_with_healing(self, task_def)
```

**Enhanced Features:**
- Automatic healing integration in task execution
- Comprehensive error handling with fallback strategies
- Healing history tracking and reporting
- Backwards compatibility with V04 patterns

## Usage Examples

### Basic Usage
```python
from src.pipeline.pipeline_executor_v05 import PipelineExecutorV05

# Initialize executor
executor = PipelineExecutorV05('config/function_registry.json')

# Run DAG with automatic healing
result = executor.run_dag_with_healing(
    dag_file='path/to/dag.json',
    user_input='Create generators with 100MW capacity',
    initial_context={'project_name': 'MyProject'}
)

# Check healing history
healing_history = executor.argument_healer.get_healing_history()
for attempt in healing_history:
    print(f"Task {attempt['task_id']}: {attempt['status']}")
```

### Direct Healing Usage
```python
from src.pipeline.pipeline_executor_v05 import ArgumentHealer

# Create healer instance
healer = ArgumentHealer(function_registry, module_map)

# Heal specific arguments
healed_inputs = healer.heal_arguments(
    user_input="Create generator",
    task_outputs=previous_outputs,
    current_inputs=failed_inputs,
    original_inputs=task_specification,
    task_def=task_definition,
    task_id="create_gen_001"
)
```

## Common Healing Scenarios

### 1. Missing Reference Resolution
**Problem**: `task_outputs.missing_task.value` → `None`
**Solution**: Search all task outputs for similar values using fuzzy matching

### 2. None Value Substitution
**Problem**: Required argument is `None`
**Solution**: Find appropriate values from context or execute data retrieval functions

### 3. Malformed Arguments
**Problem**: Incorrect data types or format
**Solution**: LLM analyzes requirements and suggests proper formatting

### 4. Context Dependency Issues
**Problem**: Arguments depend on data not yet available
**Solution**: Trigger additional function calls to gather required information

## Integration with V04

The V05 system maintains backwards compatibility while enhancing V04 with basic healing:

```python
# V04 can import basic healing function
from .pipeline_executor_v05 import argument_healing

# Basic healing in V04 exception handler
try:
    result = func_to_call(**inputs)
except Exception as e:
    healed_inputs = argument_healing(user_input, task_outputs, inputs, 
                                   original_inputs, task_def, task_id)
    result = func_to_call(**healed_inputs)
```

## Configuration

### Function Registry Integration
The healing system uses the existing `function_registry.json` to:
- Identify available functions for data resolution
- Provide LLM with context about capabilities
- Execute suggested healing functions automatically

### Healing Parameters
- `max_attempts`: Maximum healing attempts per task (default: 2)
- `function_timeout`: Timeout for healing function execution
- `llm_context_limit`: Maximum context size for LLM healing requests

## Testing

Run the test suite to verify healing functionality:

```bash
python test_argument_healing.py
```

This will test:
- DAG context healing scenarios
- LLM-based resolution
- Function registry integration
- Healing history tracking
- V04/V05 compatibility

## Benefits

### 🚀 Improved Reliability
- Reduces pipeline failures due to argument issues
- Automatically recovers from common data flow problems
- Maintains execution continuity even with partial failures

### 🧠 Intelligent Recovery
- Uses AI to understand argument requirements
- Learns from DAG context and user intent
- Suggests and executes appropriate resolution strategies

### 📊 Enhanced Debugging
- Detailed healing history for troubleshooting
- Clear error messages and resolution attempts
- Traceable argument transformation process

### 🔧 Maintainability
- Reduces need for manual pipeline fixing
- Self-healing capabilities reduce support overhead
- Comprehensive logging for issue analysis

## Future Enhancements

- **Machine Learning**: Learn from successful healing patterns
- **Advanced Context**: Integration with external knowledge bases
- **Predictive Healing**: Prevent failures before they occur
- **Custom Healing**: User-defined healing strategies
- **Performance Optimization**: Caching and optimization for common patterns

---

*This argument healing system represents a significant advancement in pipeline reliability and intelligence, making PLEXOS modeling workflows more robust and user-friendly.*
