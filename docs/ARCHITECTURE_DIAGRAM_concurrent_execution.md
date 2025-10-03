# Concurrent Execution Architecture Diagram

## Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Report Generation Pipeline                    │
│                         pipeline.run()                           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │  Check Concurrency    │
                    │  Config Flags         │
                    └───────────┬───────────┘
                                │
        ┌───────────────────────┴────────────────────────┐
        │                                                 │
        ▼                                                 ▼
┌───────────────────────┐                   ┌─────────────────────────┐
│  SEQUENTIAL MODE      │                   │  CONCURRENT MODE        │
│  (Debugging)          │                   │  (Production)           │
└───────────────────────┘                   └─────────────────────────┘
        │                                                 │
        │ For each objective:                            │
        │   process_objective()                          │ ThreadPoolExecutor
        │                                                 │ (max_workers=N)
        ▼                                                 ▼
┌───────────────────────┐                   ┌─────────────────────────┐
│  Objective 1 → 2 → 3  │                   │  Obj 1 ║ Obj 2 ║ Obj 3  │
│  (One at a time)      │                   │  (All in parallel)      │
└───────────┬───────────┘                   └────────────┬────────────┘
            │                                             │
            │                                             │ Each Objective:
            │                                             │
            ▼                                             ▼
    ┌───────────────┐                           ┌────────────────────┐
    │ parse_objectives()                        │ _process_single_   │
    │                                           │ objective()        │
    └───────────────┘                           └────────────────────┘
            │                                             │
            │ Check run_task_concurrent                   │
            │                                             │
    ┌───────┴───────┐                           ┌────────┴────────┐
    │               │                           │                 │
    ▼               ▼                           ▼                 ▼
┌──────────┐  ┌──────────┐            ┌──────────┐      ┌──────────┐
│Sequential│  │Concurrent│            │Sequential│      │Concurrent│
│Tasks     │  │Tasks     │            │Tasks     │      │Tasks     │
└──────────┘  └──────────┘            └──────────┘      └──────────┘
    │               │                       │                 │
    │               │ ThreadPoolExecutor    │                 │
    │               │ (tasks)               │                 │
    ▼               ▼                       ▼                 ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│  Task 1 → Task 2 → Task 3   │   │  Task 1 ║ Task 2 ║ Task 3  │
└─────────────┬───────────────┘   └───────────┬─────────────────┘
              │                               │
              ▼                               ▼
      ┌───────────────┐               ┌───────────────┐
      │ Sub-tasks     │               │ Sub-tasks     │
      │ (Sequential)  │               │ (Sequential)  │
      └───────────────┘               └───────────────┘
              │                               │
              ▼                               ▼
    ┌─────────────────────┐         ┌─────────────────────┐
    │ Report Structure    │         │ Report Structure    │
    │ (Thread-safe lock)  │         │ (Thread-safe lock)  │
    └─────────────────────┘         └─────────────────────┘
```

## Thread Safety Architecture

```
┌────────────────────────────────────────────────────────────┐
│                  ReportStructureManager                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │           self.report_structure: Dict               │  │
│  │              (Shared Resource)                      │  │
│  └─────────────────────────────────────────────────────┘  │
│                          │                                 │
│                          │ Protected by                    │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              self._lock: Lock                       │  │
│  │            (Threading Lock)                         │  │
│  └─────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘

Thread 1                    Thread 2                    Thread 3
────────                    ────────                    ────────
    │                           │                           │
    │ Refine text               │ Refine text               │ Refine text
    │ (outside lock)            │ (outside lock)            │ (outside lock)
    │                           │                           │
    ▼                           ▼                           ▼
┌────────┐                 ┌────────┐                 ┌────────┐
│ LLM    │                 │ LLM    │                 │ LLM    │
│ Call   │                 │ Call   │                 │ Call   │
└───┬────┘                 └───┬────┘                 └───┬────┘
    │                           │                           │
    │ with self._lock:          │ with self._lock:          │ with self._lock:
    ▼                           ▼                           │ (waits...)
┌────────────┐             ┌────────────┐                  │
│ ACQUIRED   │             │  WAITING   │                  │
│ Update     │             │            │                  │
│ structure  │             │            │                  │
└─────┬──────┘             └─────┬──────┘                  │
      │                          │                          │
      │ Release lock             │ ACQUIRED                 │
      └─────────────────────────>│ Update                   │
                                 │ structure                │
                                 └────┬─────────────────────┘
                                      │ Release lock
                                      └────────────────────>│ ACQUIRED
                                                            │ Update
                                                            │ structure
                                                            └─────────>
```

## Configuration Decision Tree

```
                    ┌─────────────────────┐
                    │  Start Pipeline     │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴─────────────────────┐
                    │ run_objectives_concurrent?     │
                    └──────────┬─────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
            ┌───────┐                    ┌──────────┐
            │ FALSE │                    │   TRUE   │
            └───┬───┘                    └─────┬────┘
                │                              │
                │ Sequential                   │ ThreadPoolExecutor
                │ Objectives                   │ (Parallel Objectives)
                │                              │
                ▼                              ▼
    ┌──────────────────────┐      ┌──────────────────────┐
    │ For each objective:  │      │ Submit all objectives│
    │   parse_objectives() │      │ to thread pool       │
    └──────────┬───────────┘      └──────────┬───────────┘
               │                              │
               │                              │
               └──────────────┬───────────────┘
                              │
                  ┌───────────┴─────────────────┐
                  │  run_task_concurrent?       │
                  └───────────┬─────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
            ┌───────┐                  ┌──────────┐
            │ FALSE │                  │   TRUE   │
            └───┬───┘                  └─────┬────┘
                │                            │
                │ Sequential                 │ ThreadPoolExecutor
                │ Tasks                      │ (Parallel Tasks)
                │                            │
                ▼                            ▼
    ┌────────────────────┐      ┌────────────────────────┐
    │ For each task:     │      │ Submit all tasks       │
    │   process_task()   │      │ to thread pool         │
    └────────────────────┘      └────────────────────────┘
```

## Performance Comparison

```
Sequential Mode (Both Flags = False)
═══════════════════════════════════
Time: ████████████████████████████████████████ 180 min
      └─ Obj1 ─┘└─ Obj2 ─┘└─ Obj3 ─┘└─ Obj4 ─┘└─ Obj5 ─┘

Chapter Parallel (run_objectives_concurrent = True)
═══════════════════════════════════════════════════
Time: ████████████████ 50 min
      ║ Obj1 ║ Obj2 ║ Obj3 ║ Obj4 ║ Obj5 ║

Task Parallel (run_task_concurrent = True)
═══════════════════════════════════════════
Time: ████████████████████████ 75 min
      └─ Obj1(║T1║T2║T3║) ─┘└─ Obj2(║T1║T2║) ─┘...

Full Parallel (Both Flags = True)
═══════════════════════════════════
Time: ████████ 25 min
      ║ Obj1(║T1║T2║T3║) ║ Obj2(║T1║T2║) ║ Obj3... ║

Legend:
└─ ─┘  Sequential
║   ║  Concurrent
```

## Memory Usage Pattern

```
Sequential Mode
───────────────
Memory │     ┌──┐     ┌──┐     ┌──┐
Usage  │     │  │     │  │     │  │
       │  ┌──┘  └──┐──┘  └──┐──┘  └──┐
       └──┴─────────┴─────────┴─────────┴──> Time
          Low, consistent memory usage

Concurrent Mode
───────────────
Memory │  ┌────────────────┐
Usage  │  │                │
       │  │  All objectives│
       │  │  in memory     │
       └──┴────────────────┴──────────> Time
          Higher peak, faster completion
```

## Lock Contention Analysis

```
Low Contention (Good Design)
────────────────────────────

Thread 1: ████ LLM Call ████│L│
Thread 2:     ████ LLM Call ████│L│
Thread 3:         ████ LLM Call ████│L│
Thread 4:             ████ LLM Call ████│L│

L = Lock acquisition (very brief)
Most time spent outside lock


High Contention (Bad Design - Avoided)
───────────────────────────────────────

Thread 1: │──────── LOCKED ────────│
Thread 2: │ wait ──│──── LOCKED ────│
Thread 3: │ wait ──────│── LOCKED ──│
Thread 4: │ wait ──────────│ LOCKED │

All operations inside lock = serialized
```

## Error Handling Flow

```
┌────────────────────────────────────┐
│  ThreadPoolExecutor.submit()       │
│  Multiple tasks submitted          │
└────────────┬───────────────────────┘
             │
             │ Task 1, 2, 3, 4, 5 running
             │
    ┌────────┼────────┬────────┬────────┐
    │        │        │        │        │
    ▼        ▼        ▼        ▼        ▼
┌───────┐┌───────┐┌───────┐┌───────┐┌───────┐
│Task 1 ││Task 2 ││Task 3 ││Task 4 ││Task 5 │
│  ✓    ││  ✗    ││  ✓    ││  ✓    ││  ✓    │
└───┬───┘└───┬───┘└───┬───┘└───┬───┘└───┬───┘
    │        │        │        │        │
    │        │ Exception       │        │
    │        │ raised          │        │
    │        │                 │        │
    └────────┴─────────┬───────┴────────┘
                       │
            ┌──────────┴──────────┐
            │ concurrent.futures  │
            │ .wait(futures)      │
            └──────────┬──────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │ Check each future:   │
            │  future.result()     │
            └──────────┬───────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
    ┌───────┐    ┌─────────┐    ┌───────┐
    │Success│    │Exception│    │Success│
    │ (Log) │    │(Log err)│    │ (Log) │
    └───────┘    └─────────┘    └───────┘
                       │
                       │ Continue execution
                       ▼
            ┌──────────────────────┐
            │ Partial results      │
            │ still available      │
            └──────────────────────┘
```

## Summary

This architecture provides:
- ✅ **Scalability**: Automatic scaling to workload
- ✅ **Safety**: Thread-safe operations with locks
- ✅ **Flexibility**: Configurable at multiple levels
- ✅ **Reliability**: Graceful error handling
- ✅ **Performance**: Significant speedup (4-12x)
- ✅ **Observability**: Clear logging and monitoring
