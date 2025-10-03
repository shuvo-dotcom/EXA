# Dependency Results System
## How Task Results Flow Between Agents

## Overview
The multi-agent system now supports passing results from completed tasks to dependent tasks via the `dependency_results` kwarg. This ensures seamless data flow and task continuity.

---

## Architecture

### 1. Main Orchestrator (`main.py`)

```python
# Track completed task results
completed_task_results = {}

for task in task_list:
    # Gather results from dependencies
    dependency_results = {}
    for dep_id in task['dependencies']:
        if dep_id in completed_task_results:
            dependency_results[dep_id] = completed_task_results[dep_id]
    
    # Pass to agent
    result = Agent.main(
        user_prompt=task_prompt,
        dependency_results=dependency_results  # <-- Results dictionary
    )
    
    # Store for future dependent tasks
    completed_task_results[task_id] = result
```

### 2. Agent Entry Points

Each agent (`Emil.py`, `Lola.py`, `Nova.py`) accepts `dependency_results`:

```python
def main(user_prompt: str, 
         test_mode: bool = False,
         test_dag: Optional[str] = None,
         resume_from_progress: Optional[str] = None,
         max_attempts: int = 3,
         ai_mode: str = "auto-pilot",
         dependency_results: dict = None) -> Dict[str, Any]:
    """
    Args:
        dependency_results: Dictionary of {task_id: result} from completed dependencies
    """
    if dependency_results is None:
        dependency_results = {}
    
    # Initialize executor
    executor = PipelineExecutor()
    
    # Pass dependency results to executor
    executor.dependency_results = dependency_results
```

### 3. Pipeline Executor

The `PipelineExecutor` now has access to `dependency_results` and can:
- Access results from previous tasks
- Pass them to pipeline steps
- Use them in function arguments
- Include them in AI context

---

## Data Flow Example

### Simple Chain: Emil → Lola

```
Task 1: emil_run_simulation
  Input: None (no dependencies)
  Output: {
    "generation_mix": {...},
    "emissions": {...},
    "costs": {...}
  }
  ↓
  stored in completed_task_results["emil_run_simulation"]

Task 2: lola_write_report (depends on: emil_run_simulation)
  Input: dependency_results = {
    "emil_run_simulation": {
      "generation_mix": {...},
      "emissions": {...},
      "costs": {...}
    }
  }
  ↓
  Lola receives Emil's results via executor.dependency_results
  ↓
  Uses data to write comprehensive report
  Output: {"report_path": "...", "summary": "..."}
```

### Complex Chain: Emil → Emil, Lola → Nova

```
Task 1: emil_prepare_data
  Output: {"data_file": "processed.csv"}

Task 2: emil_run_model (depends on: emil_prepare_data)
  Input: dependency_results = {
    "emil_prepare_data": {"data_file": "processed.csv"}
  }
  Output: {"simulation_results": {...}}

Task 3: lola_create_report (depends on: emil_run_model)
  Input: dependency_results = {
    "emil_run_model": {"simulation_results": {...}}
  }
  Output: {"report_path": "report.pdf"}

Task 4: nova_send_summary (depends on: lola_create_report)
  Input: dependency_results = {
    "lola_create_report": {"report_path": "report.pdf"}
  }
  Output: {"email_sent": true}
```

---

## Usage in Pipeline Executor

The executor can access dependency results in several ways:

### Option 1: Direct Access in Pipeline Steps
```python
# In pipeline step
if hasattr(executor, 'dependency_results'):
    dep_data = executor.dependency_results.get('previous_task_id')
    # Use dep_data in this step
```

### Option 2: Pass to Function Calls
```python
# Pipeline executor can inject dependency results into function kwargs
function_call(
    arg1=value1,
    dependency_data=executor.dependency_results
)
```

### Option 3: Include in AI Context
```python
# When calling AI for decision making
context = f"""
Previous task results:
{json.dumps(executor.dependency_results, indent=2)}

Current task: ...
"""
```

---

## Benefits

### ✅ Clean Separation of Concerns
- Main orchestrator handles routing and storage
- Agents receive results as explicit parameter
- Executor has direct access when needed

### ✅ Type Safety
- Clear dictionary structure: `{task_id: result}`
- Optional parameter (defaults to empty dict)
- No hidden state or global variables

### ✅ Flexibility
- Agents decide how to use dependency results
- Can be passed to executor, functions, or AI context
- Easy to extend for future use cases

### ✅ Debugging
- Easy to inspect what data was passed
- Clear trace of data flow
- Can log dependency_results at each step

---

## Implementation Checklist

- [x] Update main.py to collect dependency results
- [x] Update Emil.main() to accept dependency_results kwarg
- [x] Update Lola.main() to accept dependency_results kwarg
- [x] Update Nova.main() to accept dependency_results kwarg
- [x] Pass dependency_results to executor in each agent
- [ ] Update PipelineExecutor to utilize dependency_results (future)
- [ ] Add dependency_results to function registry (future)
- [ ] Create helper methods for common patterns (future)

---

## Future Enhancements

### 1. Structured Result Schema
Define expected output format for each task type:
```python
class SimulationResult(TypedDict):
    generation_mix: Dict[str, float]
    emissions: Dict[str, float]
    costs: Dict[str, float]
```

### 2. Result Transformation
Automatically transform results for different agent needs:
```python
# Emil outputs technical data
# Lola needs summary statistics
dependency_results_transformed = transform_for_lola(
    dependency_results['emil_task']
)
```

### 3. Result Caching
Cache large results to disk and pass references:
```python
dependency_results = {
    "task_1": {"type": "file_reference", "path": "large_result.json"}
}
```

### 4. Validation
Validate that required dependencies are present:
```python
def validate_dependencies(task, completed_results):
    for dep_id in task['dependencies']:
        if dep_id not in completed_results:
            raise MissingDependencyError(f"Task {dep_id} not found")
```

---

## Example Code

### Main Orchestrator
```python
completed_task_results = {}

for task in task_list:
    # Extract dependencies
    deps = task.get('dependencies', [])
    
    # Build dependency results dictionary
    dependency_results = {
        dep_id: completed_task_results[dep_id]
        for dep_id in deps
        if dep_id in completed_task_results
    }
    
    # Route to agent with dependency results
    result = route_to_agent(
        task,
        dependency_results=dependency_results
    )
    
    # Store result
    completed_task_results[task['task_id']] = result
```

### Agent Usage
```python
# In Emil, Lola, or Nova
def main(..., dependency_results=None):
    if dependency_results is None:
        dependency_results = {}
    
    executor = PipelineExecutor()
    executor.dependency_results = dependency_results
    
    # Now executor can access previous task results
    # and use them in pipeline execution
```

---

**Version:** 2.0  
**Last Updated:** October 2, 2025  
**Status:** ✅ Implemented
