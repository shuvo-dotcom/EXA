# Multi-Agent System - Quick Reference

## 🚀 Quick Start

```python
# main.py
from src.ai.architecture.choose_or_create_dag import generate_level_0_task_list
from src.EMIL import Emil
from src.LOLA import Lola
from src.NOVA import Nova

if __name__ == "__main__":
    user_prompt = "Your task description here"
    task_list = generate_level_0_task_list(user_prompt)
    
    # System automatically routes tasks to appropriate agents
```

## 📋 Agent Selection Guide

| Task Type | Agent | Example |
|-----------|-------|---------|
| Model operations | **Emil** | "Distill Joule model to Ireland" |
| Data file CRUD | **Emil** | "Modify CSV values to 999" |
| Technical analysis | **Emil** | "Calculate capacity factors" |
| Report writing | **Lola** | "Write comparison report" |
| Content creation | **Lola** | "Create blog post about renewables" |
| Social media | **Lola** | "Draft LinkedIn announcement" |
| File summarization | **Nova** | "Summarize log files" |
| Information lookup | **Nova** | "Find all PLEXOS docs" |
| Status updates | **Nova** | "Check pipeline progress" |

## 🔧 Common Parameters

```python
# All agents support these parameters:
agent.main(
    user_prompt="Task description",
    test_mode=False,              # Use test DAG
    test_dag=None,                # Path to test DAG
    resume_from_progress=None,    # Resume from progress file
    max_attempts=3,               # Retry attempts
    ai_mode='auto-pilot'          # Operation mode
)
```

## 📝 Task List Format

```json
{
  "task_id": "emil_distill_model",
  "task_description": "Distill Joule to Ireland",
  "assistant": "Emil",
  "dependencies": [],
  "priority": "high",
  "estimated_complexity": "complex",
  "expected_outputs": ["irish_model_file"],
  "input_requirements": ["joule_model_path"],
  "scope": "Detailed scope...",
  "notes": "Additional context..."
}
```

## 🔄 Execution Flow

```
User Prompt
    ↓
generate_level_0_task_list()
    ↓
Task Loop → Emil/Lola/Nova
    ↓
Generate DAG
    ↓
Execute Pipelines
    ↓
Return Outputs
```

## ⚙️ Agent Differences

| Feature | Emil | Lola | Nova |
|---------|------|------|------|
| PLEXOS Models | ✅ | ❌ | ❌ |
| Database Ops | ✅ | ❌ | ❌ |
| File CRUD | ✅ | ⚠️ | 📖 |
| Content Gen | ⚠️ | ✅ | ⚠️ |
| AI Failure Analysis | ✅ | ❌ | ❌ |

Legend: ✅ Full Support | ⚠️ Limited | ❌ No Support | 📖 Read Only

## 📁 File Structure

```
project_root/
├── main.py                    # Main orchestrator
├── src/
│   ├── EMIL/
│   │   ├── __init__.py
│   │   └── Emil.py           # Technical agent
│   ├── LOLA/
│   │   ├── __init__.py
│   │   └── Lola.py           # Communications agent
│   ├── NOVA/
│   │   ├── __init__.py
│   │   └── Nova.py           # Virtual assistant
│   └── ai/architecture/
│       └── choose_or_create_dag.py
├── config/
│   ├── default_ai_models.yaml
│   └── function_registry.json
└── pipeline_progress/         # Auto-generated
```

## 🐛 Troubleshooting

### Import Errors
```bash
# Ensure __init__.py exists in agent directories
ls src/EMIL/__init__.py
ls src/LOLA/__init__.py
ls src/NOVA/__init__.py
```

### Task Not Routing
- Check `assistant` field is "Emil", "Lola", or "Nova"
- Case-insensitive matching is supported

### Progress Not Loading
- Verify progress file exists in `pipeline_progress/`
- Check JSON is valid

## 📊 Progress Files

```python
# List available progress
from src.EMIL.Emil import list_available_progress_files
progress_files = list_available_progress_files()

# Resume from progress
result = Emil.main(
    user_prompt="...",
    resume_from_progress="pipeline_progress/progress_123.json"
)
```

## 🎯 Example Workflows

### Simple: Technical Task
```python
user_prompt = "Create a new generator in the PLEXOS model"
# → Routes to Emil only
```

### Simple: Content Task
```python
user_prompt = "Write a blog post about wind energy"
# → Routes to Lola only
```

### Complex: Multi-Agent
```python
user_prompt = """
Run capacity expansion study and write report
"""
# → Task 1: Emil (run study)
# → Task 2: Lola (write report, depends on Task 1)
```

## 🔍 Logging

All agents log to console with format:
```
[INFO] AgentName: Message
```

Check logs in:
- Console output (real-time)
- `logs/app.log` (persistent)

## 💡 Best Practices

1. **Keep prompts clear and specific**
   - ✅ "Distill Joule model to Ireland and add nuclear plant"
   - ❌ "Do some modeling work"

2. **Let agents create sub-tasks**
   - High-level tasks only
   - Agents generate detailed DAGs

3. **Use dependencies for multi-agent tasks**
   - Lola depends on Emil for data
   - System executes in order

4. **Monitor progress files**
   - Enable resume capability
   - Track execution history

## 📚 Documentation

- **Full Architecture**: `docs/MULTI_AGENT_SYSTEM_ARCHITECTURE.md`
- **Examples**: `docs/example_prompts.py`
- **Agent Docs**: Individual agent module docstrings

## 🚨 Common Errors

| Error | Solution |
|-------|----------|
| `No tasks generated` | Check prompt clarity |
| `Unknown assistant` | Verify task list format |
| `Failed to load progress` | Check file path and JSON |
| `Import error` | Verify __init__.py files |

## 🎓 Learning Path

1. **Start with simple single-agent tasks**
   - Emil: "Create a generator"
   - Lola: "Write a blog post"
   - Nova: "Summarize logs"

2. **Progress to multi-agent workflows**
   - Emil → Lola dependencies
   - Complex task decomposition

3. **Advanced: Progress management**
   - Resume failed executions
   - Analyze attempt history

## 🔗 Related Files

- `main.py` - Entry point
- `src/ai/architecture/choose_or_create_dag.py` - Task decomposition
- `config/function_registry.json` - Available functions
- `config/default_ai_models.yaml` - AI model config

---

**Quick Help**: Run `python docs/example_prompts.py [emil|lola|nova|multi|all]` to see examples
