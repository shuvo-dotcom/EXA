import traceback
import time
import json
import os
from datetime import datetime
from turtle import st
from src.pipeline.pipeline_executor_v05 import PipelineExecutor
import logging
try:
    from src.utils.logging_setup import get_logger
    logger = get_logger("Emil")
except Exception:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
    logger = logging.getLogger("Emil")

import src.EMIL.plexos.plexos_extraction_functions_agents as pef
import src.EMIL.plexos.routing_system as rs
from src.ai.PLEXOS.modelling_system_functions import choose_plexos_xml_file
from src.ai.architecture.choose_or_create_dag import choose_or_create_dag
from src.tools.mic_stt import main as mic_stt_main

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
            return progress_file
        except Exception as e:
            return None
    
    def load_progress(self, progress_file_path):
        """Load progress from a JSON file."""
        try:
            with open(progress_file_path, 'r') as f:
                progress_data = json.load(f)
            return progress_data
        except Exception as e:
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
            return []
    
    def analyze_failure_with_ai(self, exception, pipeline_file_path, current_step=None, 
                               executor_state=None, attempt_number=1):
        """Use AI to analyze the failure and provide insights."""
        try:
            # Import AI functions
            import src.ai.open_ai_calls as oaic
            
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
            
        except Exception as e:
            print(f"Failed to cleanup old progress files: {e}")

def run_pipeline(executor, user_prompt, copy_db, load_model_decision, final_outputs, pipeline_file_path, function_registry_path, ai_mode: str = "auto-pilot"):
    """
    Run the pipeline with the given parameters.
    """
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
        
    return executor.loop_dags(pipeline_file_path, function_registry_path, ai_mode=ai_mode)

def main(user_prompt, test_mode= False, test_dag=None, resume_from_progress=None, max_attempts=3, ai_mode: str = "auto-pilot", dependency_results: dict = None):    
    # Initialize progress manager
    progress_manager = PipelineProgressManager()
    
    # Store dependency results for use in pipeline execution
    if dependency_results is None:
        dependency_results = {}
    
    # Clean up old progress files (older than 7 days)
    progress_manager.cleanup_old_progress_files()
    
    # Handle resuming from previous progress
    if resume_from_progress:
        progress_data = progress_manager.load_progress(resume_from_progress)
        if not progress_data:
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

    else:
        # Fresh start
        copy_db = None
        if user_prompt:
            model_results = rs.file_copy_option(user_prompt)
            copy_db = model_results.get("copy_db", False)
            load_model_decision = model_results.get("load_model", False)

        if test_mode:
            pipeline_file_paths = [test_dag] if test_dag else []
        else:
            # choose_or_create_dag returns (filepath, reasoning_dict)
            dag_path, reasoning = choose_or_create_dag(user_prompt)
            # Ensure pipeline_file_paths is a list of string paths
            if isinstance(dag_path, (list, tuple)):
                pipeline_file_paths = [str(p) for p in dag_path]
            else:
                pipeline_file_paths = [str(dag_path)]
            logger.info("Selected/created DAG: %s", pipeline_file_paths[0])
        
        # Initialize progress tracking variables
        completed_pipelines = []
        current_pipeline_index = 0
        final_outputs = {}

    # Only proceed if user has entered input and LLM has made a decision
    start_time = time.time()
    
    function_registry_path = r'config\function_registry.json'
    executor = PipelineExecutor()
    
    # Pass dependency results to executor
    executor.dependency_results = dependency_results
    
    # Resume from where we left off
    remaining_pipelines = pipeline_file_paths[current_pipeline_index:]
    
    # Initialize attempt history for AI analysis
    attempt_history = []
    
    try:
        for i, pipeline_file_path in enumerate(remaining_pipelines):
            actual_index = current_pipeline_index + i
            
            pipeline_completed = False
            last_exception = None
            
            for attempt in range(max_attempts):
                attempt_start_time = time.time()
                current_attempt_num = attempt + 1
                
                try:
                    # Execute the pipeline
                    logger.info("Running pipeline %s (%d/%d)", pipeline_file_path, actual_index + 1, len(pipeline_file_paths))
                    pipeline_output = run_pipeline(
                        executor, user_prompt, copy_db, load_model_decision, 
                        final_outputs, pipeline_file_path, function_registry_path
                        , ai_mode=ai_mode
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
                        copy_db=copy_db,
                        load_model_decision=load_model_decision,
                        executor_state=get_executor_state_summary(executor),
                        failure_info=None,
                        attempt_history=attempt_history
                    )
                    
                    break  # Exit the retry loop on success
                    
                except Exception as e:
                    last_exception = e
                    attempt_duration = time.time() - attempt_start_time
                    logger.exception("Pipeline failed on attempt %d for %s: %s", current_attempt_num, pipeline_file_path, e)
                    
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
            
            # Update current pipeline index for the next iteration
            current_pipeline_index = actual_index + 1
    
    except Exception as e:
        logger.exception("Pipeline execution stopped due to error: %s", e)
        return final_outputs
    
    # If we reach here, all pipelines completed successfully
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info("All pipelines completed in %.2fs", elapsed_time)
    return final_outputs

def list_available_progress_files():
    """Utility function to list and select from available progress files."""
    progress_manager = PipelineProgressManager()
    progress_files = progress_manager.list_progress_files()
    
    for i, file_info in enumerate(progress_files):
        print(f"{i+1}. {file_info['timestamp']} - {file_info['user_prompt']}")
        print(f"   Progress: {file_info['completed_pipelines']}/{file_info['total_pipelines']} pipelines")

        # Get detailed statistics
        stats = get_progress_statistics(file_info['file_path'])
        if stats:
            print(f"   Attempts: {stats['successful_attempts']}✅ / {stats['failed_attempts']}❌ (Total: {stats['total_attempts']})")
            print(f"   Avg Duration: {stats['average_attempt_duration']:.1f}s")
            if stats['failure_analysis_count'] > 0:
                print(f"   AI Analyses: {stats['failure_analysis_count']}")

        print(f"   File: {file_info['file_path']}")
        print("---")
    
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

if __name__ == '__main__':
    # mic_mode = input('Enable microphone mode? (y/n): ').strip().lower() == 'y'
    # test_mode = input('Enable test mode? (y/n): ').strip().lower() == 'y'

    clone_test = "Use the joule model. Copy the Nuclear SMR category and add a new category call 'Nuclear SMR Expansion' with expansion options for the nuclear power plants"
    transfer_test = "Transfer the Nuclear category in the Generator class to the Gas Plant Class"
    split_test = "Use the DHEM Model. Use the split operation to divide Nuclear generators in france into 2 objects one in north one in south keep the object name similar to the original"
    merge_test = "Merge the object of the Nuclear SMR and Nuclear categories into a new category called 'Nuclear All'"

    mic_mode = False
    test_mode = False
    ai_mode = 'auto-pilot'  # Options: 'auto-pilot', 'co-pilot', 'manual'

    if mic_mode:
        user_prompt = mic_stt_main(duration=20)
    else:
        if test_mode: 
            # Let the user choose a DAG file from the task_lists directory
            task_lists_dir = r'task_lists'
            candidates = []
            for root, _, files in os.walk(task_lists_dir):
                for fname in files:
                    if fname.lower().endswith('.json'):
                        candidates.append(os.path.join(root, fname))

            # Sort by modification time (newest first)
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)

            if not candidates:
                # Fallback to the previous hardcoded test dag if no files found
                test_dag = r'task_lists\2025\09\24\create_hhp_tyndp2026_hourly_profiles_dag_2025-09-24_110152.json'
            else:
                # Display a concise numbered list
                print("Select a task list file from task_lists:")
                for idx, path in enumerate(candidates[:50], start=1):
                    mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"{idx}. {os.path.relpath(path)}    ({mtime})")

                # Prompt for selection; default to the most recent if input is empty or non-interactive
                try:
                    choice = input(f"Enter number (1-{len(candidates[:50])}) [default 1]: ").strip()
                    # choice = 1
                except (EOFError, KeyboardInterrupt):
                    choice = ""

                try:
                    sel = int(choice) - 1 if choice else 0
                    if sel < 0 or sel >= len(candidates[:50]):
                        sel = 0
                except Exception:
                    sel = 0

                test_dag = os.path.normpath(candidates[sel])
            with open(test_dag, 'r') as f:
                test_dag_content = json.load(f)
            user_prompt = test_dag_content.get('user_input', '')
        else:
            # user_prompt = "add a cateogory called 'Geothermal' the in generators class, in the sectoral model Model"
            # user_prompt = """   We need to integrate Iceland into the sectoral model.
            #                     Create a region called IS00 in the electricity category, then create the IS00 node with a connection to the region.
            #                     then create the Geothermal category, with an object called IS00 Geothermal. add a membership to the IS00 node, 
            #                     and add properties using the renewable generator template.
            #              """

            # user_prompt = """   Modify the DHEM model. Use the CRUD pipeline for all except task 2, where the modify data file function should be used.
            #                     1	Get the datafile attribute for the Gas Pipeline Capacity data file for the PCIPMI infrastrucutre level (use crud pipeline)
            #                     2	Create a variation of the Gas Pipeline Capacity data file for the PCIPMI infrastrucutre level by setting all values that aren't 0 to 999. Copy the H2\Gas.
            #                     3	Create a new datafile object in the Datafile class using the Gas Pipeline category call it something relating to PCIPMI capacities Unlimited. 
            #                     4   Add the filename property with the name of the datafile just created in step 2.
            #                     5	Add a scenario called unlimited_pci_pmi_pipeline_capacities
            #                     6	Add a max flow property to each object in the H2 inteconnection and H2 Internal bottleneck categories in the gas pipeline class, 
            #                         with the datafile and scenario created in the previous step. Use the crud pipeline
            #                     7	Clone the model "DHEM_v41_2030_PCIPMI_1995" and add the new scenario. Call the model “DHEM_v41_2030_PCIPMI_1995_unlmtd_pipeline”
            #              """


            # user_prompt = """   Modify the DHEM model.
            #                     Create a variation of the Gas Pipeline Capacity data file for the PCIPMIADVANCED infrastrucutre level by setting all values that aren't 0 to 999. 
            #                     File location = C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\DHEM\National Trends\2030\H2\Gas Pipeline
            #                     use the original file Capacities_H2_FIDPCIPMIADVANCED
            #                     Save file as Capacities_H2_FIDPCIPMIADVANCED_UNLIMITED.
            #              """


            # user_prompt = """   Modify the TJ Sectoral Model. 
            #                     Create a Nuclear SMR Generators in Germany (DE01) with a capacity of 1000 MW. Clone the Nuclear SMR in Nuclear 
            #                 """
            # user_prompt = """
            #                     I need to convert my demand timeseries in a format for the tst, please run the tst conversion file. Use Hydrogen as the carrier
            #                 """    

            # user_prompt = """   Modify the TJ Sectoral Model. 
            #                     Add a Nuclear SMR Generators in Germany (DE01). Create all of the template properties. Use a max capacity of 1000 MW. 
            #                 """

            # user_prompt = """   
            #                     Modify the TJ Sectoral Model. 
            #                     Add 2 new Gas Nodes 1 called 'e-kerosene import' and the other 'e-methanol import'.    
            #                     Add 2 new Gas Fields 1 called 'e-kerosene import' and the other 'e-methanol import'.
            #                     Perform a web search/LLM call to find import terminals for kerosene and methanol into Europe by ship and yearly capacity including any information on seasonal capacities. 
            #                     Ensure we have the city name so we can link them to the hydrogen nodes by locating and reading the NUTS regions e-highway file in the nodes and lines folder.
            #                     Create pipelines to link the kerosene import nodes to the relevant hydrogen node. Make the capacity of the pipelines match the import terminal capacities.
            #                 """
            # test_dag = None

            # user_prompt = """   
            #                     Modify the TJ Sectoral Model. 
            #                     Perform a web-search to find how much e-kerosene and e-methanol can be imported into Europe in 2050, in Tj/hr.
            #                     Add 2 new Gas Nodes 1 called 'e-kerosene import' and the other 'e-methanol import'. 
            #                     Add 2 new Gas Fields 1 called 'e-kerosene import' and the other 'e-methanol import' (when the pipeline gets to adding properties, use the data from the web-search as a dependency)
            #                     Add 2 new Gas Pipelines From 'BE01' to 'e-kerosene import' and 'e-methanol import'. Use an unlimited capacity.
            #                 """
            test_dag = None

            # user_prompt = """
            #                     Please create hourly hydrogen demand profiles for the TYNDP 2026 Scenarios. Use the 'NT', 'NT_HE' and 'NT_LE' Scenarios.
            #                     Use the year 2030, 2035, 2040 and 2050. Split the nodes. Convert to tst format. 
            #                     No need to access any PLEXOS database for now
            #                 """
            
            user_prompt = """
                                Please create hourly 'Hydrogen' demand profiles for the TYNDP 2026 Scenarios for Netherlands (NL). Use the 'NT', 'NT_HE' and 'NT_LE' Scenarios.
                                Use the year 2040. Split the nodes. Convert to tst format. No need to access any PLEXOS database for now.
                            """    
            user_prompt = """
                                Build the tst workbooks for hydrogen demand.
                            """    
            
            # user_prompt = split_test

    # Example 1: Normal execution with 3 retry attempts and AI analysis (default)
    try:
        final_outputs = main(user_prompt, test_mode=test_mode, test_dag=test_dag, max_attempts=3, ai_mode=ai_mode)
        print("Execution completed successfully!")
        # print(f"Final outputs: {final_outputs}")
    except Exception as e:
        traceback.print_exc()
        print(f"Execution failed: {e}")
        
        # Example 2: List available progress files with detailed statistics
        print("\nListing available progress files with AI analysis data:")
        progress_files = list_available_progress_files()
        
        # Example 3: Get detailed statistics from the most recent progress file
        if progress_files:
            latest_progress = progress_files[0]['file_path']
            stats = get_progress_statistics(latest_progress)
            if stats:
                print(f"\nDetailed statistics from latest progress file:")
                print(f"Success rate: {stats['successful_attempts']}/{stats['total_attempts']} attempts")
                print(f"AI analyses performed: {stats['failure_analysis_count']}")
                if stats['last_failure_analysis']:
                    print(f"Last AI analysis: {stats['last_failure_analysis'][:200]}...")
    
clone_test = "Copy the Nuclear SMR category and add a new category call 'Nuclear SMR Expansion' with expansion options for the nuclear power plants"
transfer_test = "Transfer the Nuclear category in the Generator class to the Gas Plant Class"
split_test = "Split the Nuclear SMR category into 2 categories into 2 categories represent the east and the west of europe"
merge_test = "Merge the object of the Nuclear SMR and Nuclear categories into a new category called 'Nuclear All'"