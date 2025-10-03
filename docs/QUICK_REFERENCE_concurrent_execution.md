# Quick Reference: Concurrent Execution

## Enable/Disable Concurrency

```python
config = ProjectConfig(
    # ... other settings ...
    
    # Run chapters (objectives) in parallel
    run_objectives_concurrent=True,   # Default: False
    
    # Run tasks in parallel within each chapter
    run_task_concurrent=True,         # Default: False
)
```

## Execution Modes Quick Table

| Mode | Config | Speed | Use Case |
|------|--------|-------|----------|
| **Sequential** | Both `False` | 1x (slowest) | Debugging, testing |
| **Chapter Parallel** | `run_objectives_concurrent=True`<br>`run_task_concurrent=False` | 3-5x | Moderate speed, lower memory |
| **Task Parallel** | `run_objectives_concurrent=False`<br>`run_task_concurrent=True` | 2-3x | Per-chapter optimization |
| **Full Parallel** | Both `True` | 4-12x (fastest) | Production, large reports |

## Quick Examples

### Fastest Execution (Full Parallelism)

```python
config.run_objectives_concurrent = True
config.run_task_concurrent = True
```

### Balanced (Moderate Speed, Lower Memory)

```python
config.run_objectives_concurrent = True
config.run_task_concurrent = False
```

### Debug Mode (Sequential)

```python
config.run_objectives_concurrent = False
config.run_task_concurrent = False
```

## Expected Time Savings

```
Example: 5 chapters, 3 tasks each

Sequential:     180 minutes
Chapter ||:     ~50 minutes  (3.6x faster)
Task ||:        ~75 minutes  (2.4x faster)
Full ||:        ~25 minutes  (7.2x faster)
```

## Console Output Indicators

### Sequential Mode
```
Processing objective 1
Processing objective 2
Processing objective 3
```

### Concurrent Mode
```
Running 3 objectives (chapters) concurrently
Processing objective 1
Processing objective 3
Processing objective 2
```

## Common Issues & Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| API rate limits | Set `run_objectives_concurrent=False` |
| Memory errors | Set `run_task_concurrent=False` |
| Debugging needed | Set both to `False` |
| Results seem wrong | Check thread-safety, verify locks |

## Performance Tips

✅ **DO:** Use full parallelism for production reports  
✅ **DO:** Use sequential mode when debugging  
✅ **DO:** Monitor API rate limits  
✅ **DO:** Check system memory before enabling  

❌ **DON'T:** Use concurrency with rate-limited APIs without monitoring  
❌ **DON'T:** Run parallel on low-memory systems  
❌ **DON'T:** Debug with concurrency enabled  

## Check Current Settings

```python
print(f"Objectives concurrent: {config.run_objectives_concurrent}")
print(f"Tasks concurrent: {config.run_task_concurrent}")
```

## Default Configuration Location

File: `config/default_ai_models.yaml` or in code:

```python
@dataclass
class ProjectConfig:
    run_objectives_concurrent: bool = True   # Set your default
    run_task_concurrent: bool = True         # Set your default
```

## Related Documentation

- Full details: `CONCURRENT_EXECUTION_IMPLEMENTATION.md`
- JSON updates: `JSON_RESPONSE_FORMAT_UPDATE.md`
- LLM refactoring: `REFACTORING_SUMMARY_open_ai_calls.md`
