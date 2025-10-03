import traceback
import streamlit as st

import time
import json
import os
from datetime import datetime

from src.pipeline.archive.pipeline_executor_v04 import PipelineExecutor
import src.EMIL.plexos.plexos_extraction_functions_agents as pef
import src.EMIL.plexos.routing_system as rs
from src.ai.PLEXOS.modelling_system_functions import choose_plexos_xml_file
from src.ai.architecture.choose_or_create_dag import choose_or_create_dag
import src.ai.llm_calls.open_ai_calls as oaic

class PipelineProgressManager:
    """
    Manages progress saving and restoration for pipeline execution with retry attempts.
    
    Features:
    - Saves progress after every attempt (successful or failed)
    - AI-powered failure analysis using OpenAI for each failed attempt
    - Tracks detailed attempt history with timing and analysis
    - Allows resuming execution from where it left off
    - Tracks completed pipelines, current pipeline index, and execution state
    - Automatic cleanup of old progress files
    - Detailed progress statistics and failure insights
    
    Usage:
    1. Normal execution with retry and auto-save after each attempt:
       main(user_prompt, max_attempts=3)
    
    2. Resume from saved progress:
       main(user_prompt, resume_from_progress="path/to/progress_file.json")
    
    3. List available progress files with statistics:
       list_available_progress_files()
    
    4. Get detailed progress statistics:
       get_progress_statistics("path/to/progress_file.json")
    
    AI Analysis Features:
    - Analyzes each failure with context about the pipeline step
    - Provides root cause analysis and potential solutions
    - Estimates likelihood of success on retry
    - Recommends next steps for troubleshooting
    """
    
    def __init__(self, progress_dir="pipeline_progress"):
        self.progress_dir = progress_dir
        if not os.path.exists(progress_dir):
            os.makedirs(progress_dir)
    
    def _get_progress_file_path(self, user_prompt, pipeline_file_paths):
        """Generate a unique progress file path based on user prompt and pipelines."""
        # Create a simple hash of the prompt and pipelines for uniqueness
        prompt_hash = str(hash(str(user_prompt) + str(pipeline_file_paths)))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"progress_{prompt_hash}_{timestamp}.json"
        return os.path.join(self.progress_dir, filename)
    
    def save_progress(self, user_prompt, pipeline_file_paths, completed_pipelines, 
                     current_pipeline_index, current_attempt, final_outputs, 
                     copy_db, load_model_decision, executor_state=None, 
                     failure_info=None, attempt_history=None):
        """Save current progress to a JSON file."""
        progress_data = {
            "timestamp": datetime.now().isoformat(),
            "user_prompt": user_prompt,
            "pipeline_file_paths": pipeline_file_paths,
            "completed_pipelines": completed_pipelines,
            "current_pipeline_index": current_pipeline_index,
            "current_attempt": current_attempt,
            "final_outputs": final_outputs,
            "copy_db": copy_db,
            "load_model_decision": load_model_decision,
            "executor_state": executor_state or {},
            "failure_info": failure_info or {},
            "attempt_history": attempt_history or []
        }
        
        progress_file = self._get_progress_file_path(user_prompt, pipeline_file_paths)
        
        try:
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2, default=str)
            st.info(f"Progress saved to: {progress_file}")
            return progress_file
        except Exception as e:
            st.error(f"Failed to save progress: {e}")
            return None
    
    def load_progress(self, progress_file_path):
        """Load progress from a JSON file."""
        try:
            with open(progress_file_path, 'r') as f:
                progress_data = json.load(f)
            st.info(f"Progress loaded from: {progress_file_path}")
            return progress_data
        except Exception as e:
            st.error(f"Failed to load progress: {e}")
            return None
    
    def list_progress_files(self):
        """List all available progress files."""
        try:
            progress_files = []
            for filename in os.listdir(self.progress_dir):
                if filename.startswith("progress_") and filename.endswith(".json"):
                    file_path = os.path.join(self.progress_dir, filename)
                    # Try to read basic info from each file
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        progress_files.append({
                            "file_path": file_path,
                            "timestamp": data.get("timestamp", "Unknown"),
                            "user_prompt": data.get("user_prompt", "Unknown")[:100] + "...",
                            "total_pipelines": len(data.get("pipeline_file_paths", [])),
                            "completed_pipelines": len(data.get("completed_pipelines", []))
                        })
                    except:
                        continue
            return sorted(progress_files, key=lambda x: x["timestamp"], reverse=True)
        except Exception as e:
            st.error(f"Failed to list progress files: {e}")
            return []
    
    def analyze_failure_with_ai(self, exception, pipeline_file_path, current_step=None, 
                               executor_state=None, attempt_number=1):
        """Use AI to analyze the failure and provide insights."""
        try:
            # Import AI functions
            
            # Prepare failure analysis prompt
            failure_context = {
                "pipeline_file": os.path.basename(pipeline_file_path) if pipeline_file_path else "Unknown",
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "current_step": current_step,
                "attempt_number": attempt_number,
                "executor_state_summary": self._summarize_executor_state(executor_state)
            }
            
            analysis_prompt = f"""
            Analyze this pipeline execution failure and provide insights:
            
            Pipeline File: {failure_context['pipeline_file']}
            Attempt Number: {failure_context['attempt_number']}
            Exception Type: {failure_context['exception_type']}
            Exception Message: {failure_context['exception_message']}
            Current Step: {failure_context.get('current_step', 'Unknown')}
            
            Executor State Summary:
            {json.dumps(failure_context['executor_state_summary'], indent=2)}
            
            Please provide:
            1. Root cause analysis
            2. Potential solutions
            3. Likelihood of success on retry (scale 1-10)
            4. Recommended next steps
            5. Whether this appears to be a transient or persistent error
            
            Format your response as a structured analysis.
            """
            
            # Get AI analysis
            ai_analysis = oaic.simple_ai_call(analysis_prompt)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "failure_context": failure_context,
                "ai_analysis": ai_analysis,
                "analysis_successful": True
            }
            
        except Exception as ai_error:
            st.warning(f"AI failure analysis failed: {ai_error}")
            return {
                "timestamp": datetime.now().isoformat(),
                "failure_context": {
                    "pipeline_file": os.path.basename(pipeline_file_path) if pipeline_file_path else "Unknown",
                    "exception_type": type(exception).__name__,
                    "exception_message": str(exception),
                    "current_step": current_step,
                    "attempt_number": attempt_number
                },
                "ai_analysis": f"AI analysis failed: {ai_error}",
                "analysis_successful": False
            }
    
    def _summarize_executor_state(self, executor_state):
        """Create a concise summary of executor state for AI analysis."""
        if not executor_state:
            return "No executor state available"
        
        summary = {}
        
        # Extract key information
        if 'task_outputs' in executor_state:
            summary['completed_tasks'] = len(executor_state['task_outputs'])
            
        if 'final_outputs' in executor_state:
            summary['final_outputs_count'] = len(executor_state['final_outputs'])
            
        if 'dag_context' in executor_state:
            dag_context = executor_state['dag_context']
            summary['dag_context'] = {
                'task_outputs_count': len(dag_context.get('task_outputs', {})),
                'active_classes': list(dag_context.get('active_classes', [])),
                'has_plexos_model': dag_context.get('plexos_model_location') is not None
            }
        
        if 'plexos_model_location' in executor_state:
            summary['plexos_model_location'] = executor_state['plexos_model_location']
        
        if 'load_model_decision' in executor_state:
            summary['load_model_decision'] = executor_state['load_model_decision']
        
        return summary
    
    def cleanup_old_progress_files(self, days_old=7):
        """Remove progress files older than specified days."""
        try:
            cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            cleaned_count = 0
            
            for filename in os.listdir(self.progress_dir):
                if filename.startswith("progress_") and filename.endswith(".json"):
                    file_path = os.path.join(self.progress_dir, filename)
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.remove(file_path)
                        cleaned_count += 1
            
            if cleaned_count > 0:
                st.info(f"Cleaned up {cleaned_count} old progress files.")
        except Exception as e:
            st.warning(f"Failed to cleanup old progress files: {e}")

def run_pipeline(executor, user_prompt, copy_db, load_model_decision, final_outputs, pipeline_file_path, function_registry_path, status = None, ai_mode: str = "co-pilot", tabs = None):
    executor.load_model_decision = load_model_decision
    if load_model_decision:
        plexos_model_location = choose_plexos_xml_file(user_prompt)
        if copy_db == True:
            print(f"Copying PLEXOS model to a new file: {plexos_model_location}")
            plexos_model_location2 = plexos_model_location.replace('.xml', '_copy.xml')
            pef.load_plexos_xml(file_name=plexos_model_location, updated_name=plexos_model_location2, new_copy=True)
        else:
            plexos_model_location2 = plexos_model_location
        executor.plexos_model_location = plexos_model_location2
    else:
        executor.plexos_model_location = None

    return executor.loop_dags(pipeline_file_path, function_registry_path, status=status, ai_mode=ai_mode, tabs=tabs)

def main(user_prompt, test_mode=True, test_dag=None, resume_from_progress=None, max_attempts=3, ai_mode: str = "auto-pilot"):
    st.title("PLEXOS Pipeline Executor")

    # Modern status container and tabs
    top_info = st.container()
    tabs = st.tabs(["Run", "Progress", "Settings"])
    
    # Initialize progress manager
    progress_manager = PipelineProgressManager()
    
    # Clean up old progress files (older than 7 days)
    progress_manager.cleanup_old_progress_files()
    
    # Handle resuming from previous progress
    if resume_from_progress:
        progress_data = progress_manager.load_progress(resume_from_progress)
        if not progress_data:
            st.error("Failed to load progress data. Starting fresh.")
            return
        
        # Extract saved state
        user_prompt = progress_data["user_prompt"]
        pipeline_file_paths = progress_data["pipeline_file_paths"]
        completed_pipelines = progress_data["completed_pipelines"]
        current_pipeline_index = progress_data["current_pipeline_index"]
        final_outputs = progress_data["final_outputs"]
        copy_db = progress_data["copy_db"]
        load_model_decision = progress_data["load_model_decision"]
        
        # Load attempt history if available
        attempt_history = progress_data.get("attempt_history", [])
        failure_info = progress_data.get("failure_info", {})
        
        with tabs[1]:
            st.info(f"Resuming execution from pipeline {current_pipeline_index + 1}/{len(pipeline_file_paths)}")
            st.write(f"Completed pipelines: {len(completed_pipelines)}")
            st.write(f"Total attempts recorded: {len(attempt_history)}")
        
        # Display recent failure analysis if available
        if failure_info and failure_info.get("analysis_successful", False):
            with tabs[1]:
                st.info("Previous Failure Analysis:")
                st.text(failure_info.get("ai_analysis", "No analysis available"))
        
        # Show recent attempt summary
        if attempt_history:
            recent_attempts = attempt_history[-3:]  # Show last 3 attempts
            with tabs[1]:
                st.write("Recent attempts:")
                for attempt in recent_attempts:
                    status_icon = "✅" if attempt["status"] == "success" else "❌"
                    st.write(f"{status_icon} Attempt {attempt['attempt_number']} - {attempt['status']} - {attempt.get('duration_seconds', 0):.1f}s")
        
    else:
        # Fresh start
        copy_db = None
        if user_prompt:
            model_results = rs.file_copy_option(user_prompt)
            copy_db = model_results.get("copy_db", False)
            load_model_decision = model_results.get("load_model", False)
            # with tabs[0]:
            #     st.write(f"LLM decision: {'Copy the database' if copy_db else 'Do not copy the database'}")

        if test_mode:
            pipeline_file_paths = [test_dag] if test_dag else []
        else:
            pipeline_file_paths = choose_or_create_dag(user_prompt, st_tabs=tabs)

        # Initialize progress tracking variables
        completed_pipelines = []
        current_pipeline_index = 0
        final_outputs = {}

    # Only proceed if user has entered input and LLM has made a decision
    start_time = time.time()

    function_registry_path = r'config\\function_registry.json'
    executor = PipelineExecutor(function_registry_path=function_registry_path)

    # Resume from where we left off
    remaining_pipelines = pipeline_file_paths[current_pipeline_index:]

    # Initialize attempt history for AI analysis
    attempt_history = []
    
    try:
        for i, pipeline_file_path in enumerate(remaining_pipelines):
            actual_index = current_pipeline_index + i
            with tabs[2]:
                st.write(f"\n--- Processing Pipeline {actual_index + 1}/{len(pipeline_file_paths)}: {pipeline_file_path} ---")
            
            pipeline_completed = False
            last_exception = None
            
            for attempt in range(max_attempts):
                attempt_start_time = time.time()
                current_attempt_num = attempt + 1
                
                try:
                    with st.status(f"Attempt {current_attempt_num}/{max_attempts} for {os.path.basename(pipeline_file_path)}", expanded=False) as status:
                        status.update(label="Running pipeline...", state="running")
                        # Execute the pipeline
                        pipeline_output = run_pipeline(
                            executor, user_prompt, copy_db, load_model_decision, 
                            final_outputs, pipeline_file_path, function_registry_path, status=status
                            , ai_mode=ai_mode, tabs=tabs
                        )
                        status.update(label="Completed", state="complete")
                        status.update(label="Completed", state="complete")
                    
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
                        copy_db=copy_db,
                        load_model_decision=load_model_decision,
                        executor_state=get_executor_state_summary(executor),
                        failure_info=None,
                        attempt_history=attempt_history
                    )
                    
                    st.success(f"Pipeline {actual_index + 1} completed successfully!")
                    break  # Exit the retry loop on success
                    
                except Exception as e:
                    last_exception = e
                    attempt_duration = time.time() - attempt_start_time
                    
                    st.error(f"Attempt {current_attempt_num} failed: {e}")
                    
                    # Perform AI analysis of the failure
                    st.info("Analyzing failure with AI...")
                    failure_analysis = progress_manager.analyze_failure_with_ai(
                        exception=e,
                        pipeline_file_path=pipeline_file_path,
                        current_step=f"Pipeline {actual_index + 1}/{len(pipeline_file_paths)}",
                        executor_state=get_executor_state_summary(executor),
                        attempt_number=current_attempt_num
                    )
                    
                    # Record failed attempt with AI analysis
                    attempt_info = {
                        "attempt_number": current_attempt_num,
                        "pipeline_index": actual_index,
                        "pipeline_file": pipeline_file_path,
                        "status": "failed",
                        "duration_seconds": attempt_duration,
                        "timestamp": datetime.now().isoformat(),
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "ai_analysis": failure_analysis
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
                        copy_db=copy_db,
                        load_model_decision=load_model_decision,
                        executor_state=get_executor_state_summary(executor),
                        failure_info=failure_analysis,
                        attempt_history=attempt_history
                    )
                    
                    # Display AI analysis to user
                    if failure_analysis.get("analysis_successful", False):
                        with tabs[1]:
                            st.info("AI Failure Analysis:")
                            st.text(failure_analysis.get("ai_analysis", "No analysis available"))
                    
                    if current_attempt_num < max_attempts:  # If it's not the last attempt
                        st.write("Retrying after a short delay...")
                        time.sleep(5)  # Wait for 5 seconds before retrying
                    else:
                        # This was the last attempt
                        st.error(f"All {max_attempts} attempts failed for pipeline: {pipeline_file_path}")
                        
                        if failure_analysis.get("analysis_successful", False):
                            st.error("Final AI Analysis Summary:")
                            st.text(failure_analysis.get("ai_analysis", "No analysis available"))
                        
                        st.info("Progress has been saved with detailed failure analysis.")
                        st.info("You can resume later and review the AI analysis for insights.")
                        
                        # Re-raise the exception to stop execution
                        raise last_exception
            
            # Update current pipeline index for the next iteration
            current_pipeline_index = actual_index + 1
    
    except Exception as e:
        st.error(f"Pipeline execution stopped due to error: {e}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        with tabs[1]:
            st.write(f"\nExecution stopped after {elapsed_time:.2f} seconds.")
        
        # Display summary of what was completed
        with tabs[1]:
            st.write(f"\nCompleted pipelines ({len(completed_pipelines)}/{len(pipeline_file_paths)}):")
            for completed in completed_pipelines:
                st.write(f"✅ {os.path.basename(completed)}")
        
        # Show remaining pipelines
        remaining = [p for p in pipeline_file_paths if p not in completed_pipelines]
        if remaining:
            with tabs[1]:
                st.write(f"\nRemaining pipelines ({len(remaining)}):")
                for remaining_pipeline in remaining:
                    st.write(f"⏸️ {os.path.basename(remaining_pipeline)}")
        
        return final_outputs
    
    # If we reach here, all pipelines completed successfully
    end_time = time.time()
    elapsed_time = end_time - start_time
    st.success(f"All pipelines completed successfully in {elapsed_time:.2f} seconds.")
    
    # Display final summary
    with tabs[1]:
        st.write(f"\nCompleted pipelines ({len(completed_pipelines)}/{len(pipeline_file_paths)}):")
        for completed in completed_pipelines:
            st.write(f"✅ {os.path.basename(completed)}")
    
    return final_outputs

def list_available_progress_files():
    """Utility function to list and select from available progress files."""
    progress_manager = PipelineProgressManager()
    progress_files = progress_manager.list_progress_files()
    
    if not progress_files:
        st.info("No progress files found.")
        return None
    
    st.write("Available progress files:")
    for i, file_info in enumerate(progress_files):
        st.write(f"{i+1}. {file_info['timestamp']} - {file_info['user_prompt']}")
        st.write(f"   Progress: {file_info['completed_pipelines']}/{file_info['total_pipelines']} pipelines")
        st.write(f"   File: {file_info['file_path']}")
        st.write("---")
    return progress_files

def resume_pipeline_execution(progress_file_path):
    """Utility function to resume pipeline execution from a progress file."""
    return main(user_prompt="", resume_from_progress=progress_file_path)

def get_progress_statistics(progress_file_path):
    """Get detailed statistics from a progress file."""
    progress_manager = PipelineProgressManager()
    progress_data = progress_manager.load_progress(progress_file_path)
    
    if not progress_data:
        return None
    
    attempt_history = progress_data.get("attempt_history", [])
    
    stats = {
        "total_pipelines": len(progress_data.get("pipeline_file_paths", [])),
        "completed_pipelines": len(progress_data.get("completed_pipelines", [])),
        "total_attempts": len(attempt_history),
        "successful_attempts": len([a for a in attempt_history if a["status"] == "success"]),
        "failed_attempts": len([a for a in attempt_history if a["status"] == "failed"]),
        "average_attempt_duration": 0,
        "failure_analysis_count": 0,
        "last_failure_analysis": None
    }
    
    # Calculate average duration
    if attempt_history:
        total_duration = sum(a.get("duration_seconds", 0) for a in attempt_history)
        stats["average_attempt_duration"] = total_duration / len(attempt_history)
        
        # Count AI analyses
        stats["failure_analysis_count"] = len([a for a in attempt_history 
                                             if a.get("ai_analysis", {}).get("analysis_successful", False)])
        
        # Get last failure analysis
        failed_attempts = [a for a in attempt_history if a["status"] == "failed"]
        if failed_attempts:
            last_failed = failed_attempts[-1]
            if last_failed.get("ai_analysis", {}).get("analysis_successful", False):
                stats["last_failure_analysis"] = last_failed["ai_analysis"]["ai_analysis"]
    
    return stats

def get_executor_state_summary(executor):
    """Get a summary of the executor's current state for progress saving."""
    state_summary = {}
    
    # Capture important executor attributes
    if hasattr(executor, 'task_outputs'):
        state_summary['task_outputs'] = getattr(executor, 'task_outputs', {})
    
    if hasattr(executor, 'final_outputs'):
        state_summary['final_outputs'] = getattr(executor, 'final_outputs', {})
    
    if hasattr(executor, 'dag_context'):
        state_summary['dag_context'] = getattr(executor, 'dag_context', {})
    
    if hasattr(executor, 'plexos_model_location'):
        state_summary['plexos_model_location'] = getattr(executor, 'plexos_model_location', None)
    
    if hasattr(executor, 'load_model_decision'):
        state_summary['load_model_decision'] = getattr(executor, 'load_model_decision', False)
    
    return state_summary

def app():
    # Page configuration
    st.set_page_config(page_title="PLEXOS Pipeline Executor", layout="wide", initial_sidebar_state="expanded")
    st.sidebar.title("Pipeline Runner")
    st.sidebar.write("Configure and run pipelines")

    # Session state for persistent runs across reruns
    if "run_pipelines" not in st.session_state:
        st.session_state["run_pipelines"] = False
    if "run_params" not in st.session_state:
        st.session_state["run_params"] = {}

    test_mode = st.sidebar.checkbox("Test mode", value=True, help="Use a specific DAG file instead of choosing via UI")
    test_dag = None
    if test_mode:
        test_dag_path = r'task_lists\2025\08\31\tj_sectoral_model_import_nodes_and_kerosene_pipeline_dag_2025-08-31_005114.json'

        test_dag = st.sidebar.text_input("Test DAG path", value=test_dag_path, placeholder=r"task_lists\\YYYY\\MM\\DD\\your_dag.json") or None
        user_input = """                            
                            Modify the TJ Sectoral Model. 
                            Add 2 new Gas Nodes 1 called 'e-kerosene import' and the other 'e-methanol import'.    
                            Add 2 new Gas Fields 1 called 'e-kerosene import' and the other 'e-methanol import'.
                            Perform a web search/LLM call to find import terminals for kerosene and methanol into Europe by ship and yearly capacity including any information on seasonal capacities. 
                            Ensure we have the city name so we can link them to the hydrogen nodes by locating and reading the NUTS regions e-highway file in the nodes and lines folder.
                            Create pipelines to link the kerosene import nodes to the relevant hydrogen node. Make the capacity of the pipelines match the import terminal capacities.

                    """

    else:
        user_input = ""

    # Inputs
    user_prompt = st.sidebar.text_area("User prompt", value=user_input, height=120, placeholder="Describe the change/task for the pipeline...")
    # AI mode toggle: False -> co-pilot, True -> auto-pilot

    ai_mode_toggle = st.sidebar.checkbox(
                                            "AI mode: Auto-pilot",
                                            value=False,
                                            help="Toggle between Co-pilot (off) and Auto-pilot (on)"
                                        )
    ai_mode = "auto-pilot" if ai_mode_toggle else "co-pilot"

    resume_from_progress = st.sidebar.text_input("Resume from progress file (optional)", value="") or None
    max_attempts = st.sidebar.number_input("Max attempts", min_value=1, max_value=10, value=3)

    if st.sidebar.button("Run pipelines", type="primary"):
        # Persist parameters and mark as running, then rerun to render main continuously
        st.session_state["run_pipelines"] = True
        st.session_state["run_params"] = {
                                            "user_prompt": user_prompt,
                                            "test_mode": test_mode,
                                            "test_dag": test_dag,
                                            "resume_from_progress": resume_from_progress,
                                            "max_attempts": int(max_attempts),
                                            "ai_mode": ai_mode,
                                        }
        st.rerun()

    # Optional stop control
    if st.sidebar.button("Stop"):
        st.session_state["run_pipelines"] = False

    # Helpful utilities
    with st.sidebar.expander("Progress utilities"):
        if st.button("List progress files"):
            files = list_available_progress_files()
            if files:
                st.write(files)
        selected_resume = st.text_input("Resume file path", value="")
        if st.button("Resume from file") and selected_resume:
            try:
                main(
                        user_prompt=user_input,
                        test_mode=False,
                        test_dag=None,
                        resume_from_progress=selected_resume,
                        max_attempts=int(max_attempts),
                        ai_mode=ai_mode,
                    )
            except Exception:
                st.exception(traceback.format_exc())

    # If running, invoke main with persisted parameters on every rerun
    if st.session_state.get("run_pipelines"):
        params = st.session_state.get("run_params", {})
        try:
            main(
                user_prompt=params.get("user_prompt", ""),
                test_mode=params.get("test_mode", True),
                test_dag=params.get("test_dag", None),
                resume_from_progress=params.get("resume_from_progress", None),
                max_attempts=int(params.get("max_attempts", 3)),
                ai_mode=params.get("ai_mode", "co-pilot"),
            )
        except Exception:
            st.exception(traceback.format_exc())
            st.session_state["run_pipelines"] = False

# Auto-run the app when executed by Streamlit
app()
    
### TEST Cases
# ----------------------------
# Test case 1: Basic run with test DAG
clone_test = "Copy the Nuclear SMR category and add a new category call 'Nuclear SMR Expansion' with expansion options for the nuclear power plants"
transfer_test = "Transfer the Nuclear category in the Generator class to the Gas Plant Class"
split_test = "Split the Nuclear SMR category into 2 categories into 2 categories represent the east and the west of europe"
merge_test = "Merge the object of the Nuclear SMR and Nuclear categories into a new category called 'Nuclear All'"