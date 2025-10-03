
from src.ai.architecture.choose_or_create_dag import generate_level_0_task_list
from src.EMIL import Emil
from src.LOLA import Lola
from src.NOVA import Nova

if __name__ == "__main__":
    test_mode = False
    test_dag = False
    max_attempts = 3
    ai_mode = 'auto-pilot'

    user_prompt = """
                       Get emil to run to run the convert to tst format, use hydrogen as carrier.
                """    
    
    # Generate the level 0 task list
    # Note: generate_level_0_task_list returns (task_list, reasoning, dag_name)
    task_list, reasoning, dag_name = generate_level_0_task_list(user_prompt)
    
    # Validate task_list
    if not task_list:
        print("No tasks generated from the user prompt.")
        exit(1)
    
    print(f"Generated workflow: {dag_name}")
    print(f"Coordinator reasoning: {reasoning}")
    print(f"Generated {len(task_list)} tasks to execute.\n")
    
    # Track completed task results for passing to dependent tasks
    completed_task_results = {}
    
    # Iterate through each task in the task list
    for idx, task in enumerate(task_list, 1):
        task_id = task.get("task_id", f"task_{idx}")
        task_description = task.get("task_description", "No description provided")
        assistant = task.get("assistant", "").strip()
        scope = task.get("scope", task_description)
        priority = task.get("priority", "medium")
        complexity = task.get("estimated_complexity", "unknown")
        dependencies = task.get("dependencies", [])
        expected_outputs = task.get("expected_outputs", [])
        input_requirements = task.get("input_requirements", [])
        notes = task.get("notes", "None")
        
        print(f"\n{'='*80}")
        print(f"Task {idx}/{len(task_list)}: {task_id}")
        print(f"{'='*80}")
        print(f"Description: {task_description}")
        print(f"Assigned to: {assistant}")
        print(f"Priority: {priority}")
        print(f"Complexity: {complexity}")
        if dependencies:
            print(f"Dependencies: {', '.join(dependencies)}")
        print(f"{'-'*80}")
        
        # Gather results from dependent tasks
        dependency_results = {}
        if dependencies:
            print(f"\nGathering results from dependencies...")
            for dep_id in dependencies:
                if dep_id in completed_task_results:
                    dependency_results[dep_id] = completed_task_results[dep_id]
                    print(f"  ✓ Loaded results from: {dep_id}")
                else:
                    print(f"  ⚠ WARNING: Dependency '{dep_id}' not found in completed tasks!")
        
        # Prepare the task-specific prompt
        task_prompt = f"""
                        Task ID: {task_id}
                        Description: {task_description}

                        Scope: {scope}

                        Expected Outputs: {', '.join(expected_outputs) if expected_outputs else 'Not specified'}

                        Input Requirements: {', '.join(input_requirements) if input_requirements else 'Not specified'}

                        Additional Notes: {notes}
                    """
        
        print(f"\n📋 Task prompt prepared ({len(task_prompt)} characters)")
        if dependency_results:
            print(f"   Passing results from {len(dependency_results)} dependent task(s)")
        
        # Route to the appropriate assistant
        try:
            if assistant.lower() == "emil":
                print(f"Routing to Emil...\n")
                result = Emil.main(
                    user_prompt=task_description,
                    test_mode=test_mode,
                    test_dag=test_dag,
                    max_attempts=max_attempts,
                    ai_mode=ai_mode,
                    dependency_results=dependency_results  # Pass dependency results
                )
                # Store result for dependent tasks
                completed_task_results[task_id] = result
                print(f"\n✓ Task {task_id} completed successfully by Emil.")
                
            elif assistant.lower() == "lola":
                print(f"Routing to Lola...\n")
                result = Lola.main(
                    user_prompt=task_description,
                    test_mode=test_mode,
                    test_dag=test_dag,
                    max_attempts=max_attempts,
                    ai_mode=ai_mode,
                    dependency_results=dependency_results  # Pass dependency results
                )
                # Store result for dependent tasks
                completed_task_results[task_id] = result
                print(f"\n✓ Task {task_id} completed successfully by Lola.")
                
            elif assistant.lower() == "nova":
                print(f"Routing to Nova...\n")
                result = Nova.main(
                    user_prompt=task_description,
                    test_mode=test_mode,
                    test_dag=test_dag,
                    max_attempts=max_attempts,
                    ai_mode=ai_mode,
                    dependency_results=dependency_results  # Pass dependency results
                )
                # Store result for dependent tasks
                completed_task_results[task_id] = result
                print(f"\n✓ Task {task_id} completed successfully by Nova.")
                
            else:
                print(f"⚠ WARNING: Unknown assistant '{assistant}' for task {task_id}.")
                print(f"Available assistants: Emil, Lola, Nova")
                print(f"Skipping task.\n")
                continue
            
        except Exception as e:
            print(f"\n✗ ERROR: Task {task_id} failed with exception:")
            print(f"  {str(e)}")
            import traceback
            print(f"\nTraceback:")
            traceback.print_exc()
            
            # Decide whether to continue or stop on error
            user_decision = input(f"\nContinue to next task? (y/n): ").lower()
            if user_decision != 'y':
                print("Stopping execution.")
                break
            continue
    
    print(f"\n{'='*80}")
    print("All tasks processed.")
    print(f"{'='*80}")
    print(f"\nCompleted {len(completed_task_results)} tasks:")
    for task_id in completed_task_results.keys():
        print(f"  ✓ {task_id}")
    
    # Optionally save all results to a summary file
    import json
    from datetime import datetime
    summary_file = f"task_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "workflow_name": dag_name,
                "reasoning": reasoning,
                "user_prompt": user_prompt,
                "completed_tasks": list(completed_task_results.keys()),
                "results": {k: str(v) for k, v in completed_task_results.items()}
            }, f, indent=2, default=str)
        print(f"\n📄 Results saved to: {summary_file}")
    except Exception as e:
        print(f"\n⚠ Warning: Could not save results summary: {e}")


