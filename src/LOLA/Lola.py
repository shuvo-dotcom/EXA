"""
Lola - Communications Director
================================
Manages all communications, content creation, and report writing tasks.

Responsibilities:
- Website copy and content creation
- Social media management and posts
- Long-form report writing
- Communications strategy
- Technical report coordination (works with Emil for data)
- Marketing and public relations content

Entry point with consistent interface for the main orchestrator.
"""

import traceback
import time
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

try:
    from src.utils.logging_setup import get_logger
    logger = get_logger("Lola")
except Exception:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
    logger = logging.getLogger("Lola")

from src.pipeline.pipeline_executor_v05 import PipelineExecutor
from src.ai.architecture.choose_or_create_dag import choose_or_create_dag


class LolaProgressManager:
    """
    Manages progress saving and restoration for Lola's content generation tasks.
    
    Features:
    - Saves progress after every attempt (successful or failed)
    - AI-powered failure analysis for content generation issues
    - Tracks detailed attempt history with timing and analysis
    - Allows resuming execution from where it left off
    - Automatic cleanup of old progress files
    - Detailed progress statistics and failure insights
    """
    
    def __init__(self, progress_dir="pipeline_progress"):
        self.progress_dir = progress_dir
        if not os.path.exists(progress_dir):
            os.makedirs(progress_dir)
    
    def _get_progress_file_path(self, user_prompt, pipeline_file_paths):
        """Generate a unique progress file path based on user prompt and pipelines."""
        prompt_hash = str(hash(str(user_prompt) + str(pipeline_file_paths)))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lola_progress_{prompt_hash}_{timestamp}.json"
        return os.path.join(self.progress_dir, filename)
    
    def save_progress(self, user_prompt, pipeline_file_paths, completed_pipelines, 
                     current_pipeline_index, current_attempt, final_outputs, 
                     executor_state=None, failure_info=None, attempt_history=None):
        """Save current progress to a JSON file."""
        progress_data = {
            "timestamp": datetime.now().isoformat(),
            "assistant": "Lola",
            "user_prompt": user_prompt,
            "pipeline_file_paths": pipeline_file_paths,
            "completed_pipelines": completed_pipelines,
            "current_pipeline_index": current_pipeline_index,
            "current_attempt": current_attempt,
            "final_outputs": final_outputs,
            "executor_state": executor_state or {},
            "failure_info": failure_info or {},
            "attempt_history": attempt_history or []
        }
        
        progress_file = self._get_progress_file_path(user_prompt, pipeline_file_paths)
        
        try:
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2, default=str)
            logger.info("Progress saved to %s", progress_file)
            return progress_file
        except Exception as e:
            logger.error("Failed to save progress: %s", e)
            return None
    
    def load_progress(self, progress_file_path):
        """Load progress from a JSON file."""
        try:
            with open(progress_file_path, 'r') as f:
                progress_data = json.load(f)
            logger.info("Progress loaded from %s", progress_file_path)
            return progress_data
        except Exception as e:
            logger.error("Failed to load progress: %s", e)
            return None
    
    def cleanup_old_progress_files(self, days=7):
        """Remove progress files older than specified days."""
        try:
            now = time.time()
            cutoff = now - (days * 86400)
            
            for filename in os.listdir(self.progress_dir):
                if not filename.startswith("lola_progress_"):
                    continue
                filepath = os.path.join(self.progress_dir, filename)
                if os.path.getmtime(filepath) < cutoff:
                    os.remove(filepath)
                    logger.debug("Removed old progress file: %s", filename)
        except Exception as e:
            logger.warning("Failed to cleanup old progress files: %s", e)


def get_executor_state_summary(executor: PipelineExecutor) -> Dict[str, Any]:
    """Extract relevant state information from executor."""
    return {
        "current_task": getattr(executor, 'current_task', None),
        "completed_tasks": getattr(executor, 'completed_tasks', []),
        "failed_tasks": getattr(executor, 'failed_tasks', [])
    }


def run_pipeline(executor, user_prompt, final_outputs, pipeline_file_path, 
                function_registry_path, ai_mode: str = "auto-pilot"):
    """
    Execute a single pipeline for Lola's content generation tasks.
    
    Note: Lola does NOT interact with PLEXOS models, so no model loading is needed.
    """
    logger.info("Executing Lola pipeline: %s", pipeline_file_path)
    
    # Lola doesn't need PLEXOS model operations
    executor.plexos_model_location = None
    
    return executor.loop_dags(pipeline_file_path, function_registry_path, ai_mode=ai_mode)


def main(user_prompt: str, test_mode: bool = False, test_dag: Optional[str] = None, 
         resume_from_progress: Optional[str] = None, max_attempts: int = 3, 
         ai_mode: str = "auto-pilot", dependency_results: dict = None) -> Dict[str, Any]:
    """
    Main entry point for Lola - Communications Director.
    
    Args:
        user_prompt: The task description for Lola to execute
        test_mode: If True, use test_dag instead of generating new DAG
        test_dag: Path to test DAG file (only used if test_mode=True)
        resume_from_progress: Path to progress file to resume from
        max_attempts: Maximum retry attempts for failed tasks
        ai_mode: Operation mode ('auto-pilot', 'manual', etc.)
    
    Returns:
        Dict containing final outputs from all completed pipelines
    """
    logger.info("="*80)
    logger.info("LOLA - Communications Director Starting")
    logger.info("="*80)
    logger.info("Task: %s", user_prompt)
    logger.info("Mode: %s", ai_mode)
    
    # Initialize progress manager
    progress_manager = LolaProgressManager()
    
    # Clean up old progress files (older than 7 days)
    progress_manager.cleanup_old_progress_files()
    
    # Handle resuming from previous progress
    if resume_from_progress:
        progress_data = progress_manager.load_progress(resume_from_progress)
        if not progress_data:
            logger.error("Failed to load progress, starting fresh")
            resume_from_progress = None
        else:
            # Extract saved state
            user_prompt = progress_data["user_prompt"]
            pipeline_file_paths = progress_data["pipeline_file_paths"]
            completed_pipelines = progress_data["completed_pipelines"]
            current_pipeline_index = progress_data["current_pipeline_index"]
            final_outputs = progress_data["final_outputs"]
            attempt_history = progress_data.get("attempt_history", [])
            logger.info("Resuming from pipeline %d/%d", current_pipeline_index + 1, len(pipeline_file_paths))

    if not resume_from_progress:
        # Fresh start - generate DAG for Lola's task
        if test_mode:
            pipeline_file_paths = [test_dag] if test_dag else []
            logger.info("Test mode: Using test DAG")
        else:
            # Generate or select appropriate DAG for the content task
            dag_path, reasoning = choose_or_create_dag(user_prompt)
            if isinstance(dag_path, (list, tuple)):
                pipeline_file_paths = [str(p) for p in dag_path]
            else:
                pipeline_file_paths = [str(dag_path)]
            logger.info("Selected/created DAG: %s", pipeline_file_paths[0])
        
        # Initialize progress tracking variables
        completed_pipelines = []
        current_pipeline_index = 0
        final_outputs = {}
        attempt_history = []
    
    # Start execution
    start_time = time.time()
    function_registry_path = r'config\function_registry.json'
    executor = PipelineExecutor()
    
    # Store dependency results for use in pipeline execution
    if dependency_results is None:
        dependency_results = {}
    executor.dependency_results = dependency_results
    
    # Resume from where we left off
    remaining_pipelines = pipeline_file_paths[current_pipeline_index:]
    
    try:
        for i, pipeline_file_path in enumerate(remaining_pipelines):
            actual_index = current_pipeline_index + i
            
            logger.info("\n" + "="*80)
            logger.info("Pipeline %d/%d: %s", actual_index + 1, len(pipeline_file_paths), 
                       os.path.basename(pipeline_file_path))
            logger.info("="*80)
            
            pipeline_completed = False
            last_exception = None
            
            for attempt in range(max_attempts):
                attempt_start_time = time.time()
                current_attempt_num = attempt + 1
                
                logger.info("Attempt %d/%d", current_attempt_num, max_attempts)
                
                try:
                    # Execute the pipeline
                    pipeline_output = run_pipeline(
                        executor, user_prompt, final_outputs, pipeline_file_path, 
                        function_registry_path, ai_mode=ai_mode
                    )
                    
                    # If successful, store the output and mark as completed
                    final_outputs[pipeline_file_path] = pipeline_output
                    completed_pipelines.append(pipeline_file_path)
                    pipeline_completed = True
                    
                    # Record successful attempt
                    attempt_info = {
                        "attempt_number": current_attempt_num,
                        "pipeline_index": actual_index,
                        "pipeline_file": pipeline_file_path,
                        "status": "success",
                        "duration_seconds": time.time() - attempt_start_time,
                        "timestamp": datetime.now().isoformat()
                    }
                    attempt_history.append(attempt_info)
                    
                    # Save progress after successful completion
                    progress_manager.save_progress(
                        user_prompt=user_prompt,
                        pipeline_file_paths=pipeline_file_paths,
                        completed_pipelines=completed_pipelines,
                        current_pipeline_index=actual_index,
                        current_attempt=current_attempt_num,
                        final_outputs=final_outputs,
                        executor_state=get_executor_state_summary(executor),
                        failure_info=None,
                        attempt_history=attempt_history
                    )
                    
                    logger.info("✓ Pipeline completed successfully")
                    break  # Exit the retry loop on success
                    
                except Exception as e:
                    last_exception = e
                    attempt_duration = time.time() - attempt_start_time
                    logger.exception("✗ Pipeline failed on attempt %d: %s", current_attempt_num, e)
                    
                    # Record failed attempt
                    attempt_info = {
                        "attempt_number": current_attempt_num,
                        "pipeline_index": actual_index,
                        "pipeline_file": pipeline_file_path,
                        "status": "failed",
                        "duration_seconds": attempt_duration,
                        "timestamp": datetime.now().isoformat(),
                        "exception_type": type(e).__name__,
                        "exception_message": str(e)
                    }
                    attempt_history.append(attempt_info)
                    
                    # Save progress after each attempt (successful or failed)
                    progress_manager.save_progress(
                        user_prompt=user_prompt,
                        pipeline_file_paths=pipeline_file_paths,
                        completed_pipelines=completed_pipelines,
                        current_pipeline_index=actual_index,
                        current_attempt=current_attempt_num,
                        final_outputs=final_outputs,
                        executor_state=get_executor_state_summary(executor),
                        failure_info={"exception": str(e), "type": type(e).__name__},
                        attempt_history=attempt_history
                    )
                    
                    if current_attempt_num < max_attempts:
                        logger.info("Retrying... (%d/%d)", current_attempt_num + 1, max_attempts)
                    else:
                        logger.error("Max attempts reached. Pipeline failed.")
            
            # Check if pipeline failed after all attempts
            if not pipeline_completed:
                logger.error("Pipeline %s failed after %d attempts", pipeline_file_path, max_attempts)
                if last_exception:
                    raise last_exception
            
            # Update current pipeline index for the next iteration
            current_pipeline_index = actual_index + 1
    
    except Exception as e:
        logger.exception("Pipeline execution stopped due to error: %s", e)
        return final_outputs
    
    # If we reach here, all pipelines completed successfully
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info("\n" + "="*80)
    logger.info("LOLA - All Content Generation Tasks Completed")
    logger.info("="*80)
    logger.info("Total time: %.2f seconds", elapsed_time)
    logger.info("Completed pipelines: %d/%d", len(completed_pipelines), len(pipeline_file_paths))
    
    return final_outputs


if __name__ == "__main__":
    # Example usage
    test_prompt = """
                    Write a blog post about the benefits of renewable energy integration in 
                    modern power systems. Include sections on solar, wind, and storage technologies.
                    Target audience: general public. Tone: informative and optimistic.
                """
    
    result = main(
        user_prompt=test_prompt,
        test_mode=False,
        max_attempts=3,
        ai_mode='auto-pilot'
    )
    
    print("\nFinal outputs:", json.dumps(result, indent=2, default=str))
