import os
import json
import uuid
import datetime
from typing import Dict, Any, Optional, List, Tuple
import yaml
import streamlit as st
import logging

try:
    # Prefer shared project logging
    from src.utils.logging_setup import get_logger
    logger = get_logger("ChooseOrCreateDAG")
except Exception:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
    logger = logging.getLogger("ChooseOrCreateDAG")
# Support both script and module import for LLM calls


default_ai_models_file = r'config\default_ai_models.yaml'
with open(default_ai_models_file, 'r') as f:
    ai_models_config = yaml.safe_load(f)
base_model = ai_models_config.get("base_model", "gpt-5-mini")
pro_model = ai_models_config.get("pro_model", "gpt-5")
    
# Paths
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_TASK_DIR = os.path.join('task_lists')
_INDEX_FILE = os.path.join(_TASK_DIR, 'index.json')
# Path to pipeline definitions index
_PIPELINE_INDEX_FILE = os.path.join(_ROOT, 'pipeline', 'pipelines', 'index.json')

from src.ai.llm_calls.open_ai_calls import run_open_ai_ns

def load_index() -> Dict[str, Any]:
    """Load the DAG registry index from JSON."""
    if not os.path.exists(_INDEX_FILE):
        os.makedirs(os.path.dirname(_INDEX_FILE), exist_ok=True)
        with open(_INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump({'dags': []}, f, indent=2)
    
    try:
        with open(_INDEX_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content:
                return {'dags': []}
            return json.loads(content)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("Index file read failed or empty; using default. Error: %s", e)
        return {'dags': []}

def save_index(index: Dict[str, Any]):
    """Save the DAG registry index to JSON."""
    with open(_INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)

def list_existing_dags() -> List[Dict[str, Any]]:
    """Return list of existing DAG metadata entries."""
    idx = load_index()
    return idx.get('dags', [])

def select_existing_dag(user_prompt: str, dags: List[Dict[str, Any]]) -> Optional[str]:
    """Use LLM to pick an existing DAG by pipeline_id or filename."""
    if not dags:
        return None
    # Prepare prompt
    choices = '\n'.join(
        f"{i+1}. {d['pipeline_id']} - {d.get('description','')}"
        for i, d in enumerate(dags)
    )
    prompt = f"""
                User scenario: {user_prompt}
                Existing DAGs:
                {choices}
                Choose one DAG to reuse by pipeline_id or filename, or respond 'new' to create a new one.
            """
    context = "You are an assistant that selects whether to reuse an existing DAG or create a new one based on the user scenario."
    response = run_open_ai_ns(prompt, context).strip().lower()
    if response == 'new':
        return None
    # Match response to a DAG
    for d in dags:
        if response in (d['pipeline_id'].lower(), d['filename'].lower()):
            return d['filename']
    return None

def load_pipeline_index() -> List[Dict[str, str]]:
    """Load available pipelines from pipelines/index.json."""
    with open(_PIPELINE_INDEX_FILE, 'r', encoding='utf-8') as f:
        return json.load(f).get('pipelines', [])

def generate_level_0_task_list(user_prompt: str) -> Tuple[List[Dict[str, str]], str]:
    """Use LLM RAG pattern to break down the scenario into a list of tasks, returning tasks and raw reasoning."""
    # Updated instruction to only extract tasks explicitly requested by the user
    pipeline_choices, function_choices, agentic_functions = load_function_pipeline_index()

    logger.info('Creating task list...')
    date_str = datetime.date.today().isoformat()
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    prompt = f"""
        You are a high-level task coordinator that delegates work to three specialized agents: Emil, Lola, and Nova.
        
        **Agent Capabilities:**
        
        **Emil - Director of Engineering:**
        - Manages technical agents for data file operations (CSV, Excel, JSON, databases)
        - PLEXOS model interactions and modifications
        - Energy system modeling (supply/demand, infrastructure)
        - Climate modeling and hydraulic modeling
        - Network modeling and simulations
        - Technical analysis and computations
        - Model distillation and geographic subsetting
        - Functions for creating excel workbooks and csv files.
        
        **Nova - Virtual Assistant:**
        - Calendar management and scheduling
        - Email management and communications
        - Reading and summarizing files and log files
        - Providing status updates and general information
        - Administrative tasks and personal assistant duties
        - Information retrieval and lookups
        
        **Lola - Communications Director:**
        - Website copy and content creation
        - Social media management and posts
        - Long-form report writing
        - Communications strategy
        - For technical reports: coordinates with Emil via questionnaire to gather model outputs, context, and chart recommendations
        - Marketing and public relations content
        
        **Your Task:**
        Break down the following user request into a sequence of delegated tasks. Each task should be assigned to ONE agent.
        
        **Important Guidelines:**
        - Assign tasks to the most appropriate agent based on their expertise
        - Create dependencies when one task's output is needed for another (e.g., Lola depends on Emil for technical data)
        - Keep tasks HIGH-LEVEL - agents will create their own detailed task lists/DAGs
        - If Lola needs technical information, create a task for Emil first, then make Lola's task dependent on it
        - Order tasks logically based on dependencies
        - Include rich metadata to help agents understand context
        - You cannot chain tasks to the same agent, instead create a single task for that assistant to create their own DAG, only create additional tasks
            if it needs to be carried out by a different assistant.
        - You need not create a delivery or handover task, the route system will handle interations with the user.

        We want to avoid a task list which is uncessarily long, e.g. suggesting 2 seperate task which could be covered in 1 function call or pipeline.
        Here is a list of function, which you can use to guide your choices:   
        Pipelines are the PLEXOS MCP and will allow the modification of anything in the PLEXOS database. Here are some available pipelines to guide your choices:
        {pipeline_choices}

        Functions are useful for non-PLEXOS operations e.g., datafile CRUD actions, internet searches, LLM searches, etc. Here are some available functions to guide your choices:
        {function_choices}

        Return a JSON object with the following structure:
        {{
          "coordinator_reasoning": "Explanation of how you broke down the request and why you assigned tasks to specific agents",
          "tasks": [
            {{
              "task_id": "unique_task_identifier",
              "task_description": "High-level description - agent will create detailed subtasks",
              "assistant": "Emil|Nova|Lola",
              "dependencies": ["task_id_1", "task_id_2"],
              "priority": "high|medium|low",
              "estimated_complexity": "simple|moderate|complex",
              "expected_outputs": ["output_type_1", "output_type_2"],
              "input_requirements": ["what this task needs from dependencies"],
              "scope": "Detailed scope and boundaries for the agent",
              "notes": "Any additional context or instructions for the agent",
              "reasoning": "Why this task was assigned to this agent and how it fits into the overall workflow"
            }}
          ],
          "workflow_name": "descriptive_name_for_this_workflow"
        }}
        
        User Request: {user_prompt}
        
        **Example Output:**
        {{
          "coordinator_reasoning": "The user wants to distill the Joule model to Irish geography, add nuclear capacity, and compare results. This requires Emil to perform model engineering (distillation and modification), then Lola to analyze outputs and write a comparative report.",
          "tasks": [
            {{
              "task_id": "emil_distill_and_modify_model",
              "task_description": "Distill the Joule model to Irish model scope and add a nuclear power plant infrastructure",
              "assistant": "Emil",
              "dependencies": [],
              "priority": "high",
              "estimated_complexity": "complex",
              "expected_outputs": ["irish_model_file", "basecase_scenario", "nuclear_scenario", "simulation_results", "output_metrics"],
              "input_requirements": ["joule_model_path", "irish_geographic_boundaries"],
              "scope": "Geographically filter Joule model to Ireland. Add nuclear power plant with appropriate capacity, connection to grid, and technical parameters. Run basecase (without nuclear) and scenario (with nuclear) simulations. Extract key metrics: generation mix, emissions, costs, dispatch patterns, capacity factors.",
              "notes": "Emil will create his own DAG for this complex modeling task. Ensure both basecase and nuclear scenarios are run for comparison. Document all assumptions.",
              "reasoning": "Emil is the technical expert for PLEXOS model modifications and simulations. This task requires deep knowledge of energy modeling and PLEXOS capabilities."
            }},
            {{
              "task_id": "lola_write_comparison_report",
              "task_description": "Write a comprehensive report comparing the Irish basecase model outputs to the nuclear scenario",
              "assistant": "Lola",
              "dependencies": ["emil_distill_and_modify_model"],
              "priority": "high",
              "estimated_complexity": "complex",
              "expected_outputs": ["comparative_analysis_report", "executive_summary", "charts_and_visualizations"],
              "input_requirements": ["simulation_results_from_emil", "basecase_metrics", "nuclear_scenario_metrics", "chart_recommendations"],
              "scope": "Analyze differences between basecase and nuclear scenarios. Compare: generation mix changes, emission reductions, cost implications, system reliability, capacity displacement. Include executive summary, methodology, findings, and recommendations. Create/request appropriate visualizations.",
              "notes": "Lola will create her own task list for report generation. Should coordinate with Emil for technical clarifications and chart specifications. Target audience: technical stakeholders and decision-makers.",
              "reasoning": "Lola is responsible for synthesizing the findings from Emil's simulations into a coherent report. This task requires strong analytical skills and the ability to communicate complex technical information effectively."
            }}
          ],
          "workflow_name": "irish_nuclear_model_analysis_2024"
        }}
    """
    context = "You are an assistant that identifies only the explicitly requested tasks for a PLEXOS pipeline given a user scenario."
    raw = run_open_ai_ns(prompt, context, model=base_model)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.exception("Failed to parse task list JSON: %s", e)
        raise

    task_list = result.get("tasks", [])
    reasoning = result.get("reasoning", "")
    dag_name_value = result.get("dag_name", "generated_dag")
    dag_name = f"{dag_name_value}_{date_str}_{time_str.replace(':', '')}"

    logger.debug("Generated task list: %s", json.dumps(task_list, indent=2))
    logger.info("Task list reasoning: %s", reasoning)

    return task_list, reasoning, dag_name

def generate_task_list(user_prompt: str, pipeline_choices: str, function_choices: str) -> Tuple[List[Dict[str, str]], str]:
    """Use LLM RAG pattern to break down the scenario into a list of tasks, returning tasks and raw reasoning."""
    # Updated instruction to only extract tasks explicitly requested by the user
    logger.info('Creating task list...')
    date_str = datetime.date.today().isoformat()
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    prompt = f"""
        Here are some function
        Break down ONLY the tasks explicitly described in the following PLEXOS modeling scenario into a sequence of high-level tasks.
        Each task should be a single operation typically a crud action, pipeline or function call. 
        there should never be dual tasks e.g. "create and connect", create should be a task and update should be a seperate task.
        Do not introduce any additional or inferred tasks beyond what the user has requested.
        Pipelines are the PLEXOS MCP and will allow the modification of anything in the PLEXOS database. Here are some available pipelines to guide your choices:
        {pipeline_choices}

        Functions are useful for non-PLEXOS operations e.g., datafile CRUD actions, internet searches, LLM searches, etc. Here are some available functions to guide your choices:
        {function_choices}

        Use the crud_pipeline if you need to do any of these tasks. It contains specialised PLEXOS agent and DAG modifiers to perform these operations, it will carry out actions 
        on all items in sequence choosing, classes, objects, memberships, properties and attributes:
        - create: create a new model element. 
        - read: read an existing model element.
        - update: this can only be used to update an object name, memberships or property attributes of an existing model element.
        - delete: delete an existing model element.
        - clone: create a copy of an existing model element, with a new name and optionally new memberships and property attributes. Intra-class operation
        - transfer: move a model element from one location to another, updating its context and dependencies. Intra-class or cross-class operation.
        - merge: combine two or more model elements into a single element, resolving conflicts and dependencies. Intra-class operation.
        - split: divide a model element into multiple elements, preserving its structure and relationships. Intra-class operation.

        In some DAGs when modifying a PLEXOS database you may be tempted to create a membership, add_object pipeline already add default memberships, so only add additional memberships are specifically requested.
        For example if the user says something like 'create pipelines to link node a to node b' create a task to create the pipeline object but not the memberships.
        If the user says something like create a membership between abatement and generators, you should create a task for it.
        
        Return a JSON object with three keys:
        - 'tasks': a JSON array of objects with keys 'task_name' and 'description'
        - 'reasoning': a string explaining your reasoning for the task breakdown.
        - 'dag_name': a string for the DAG name.
        Scenario: {user_prompt}
        Example:
        {{
          "tasks": [
            {{
              "task_name": "create_category",
              "description": "Create a new category called 'solar csp' in the model."
            }}
          ],
          "reasoning": "The scenario requires a new category, so the first step is to create it.", 
          "dag_name": "create_solar_csp_category_dag",
        }}
    """
    context = "You are an assistant that identifies only the explicitly requested tasks for a PLEXOS pipeline given a user scenario."
    raw = run_open_ai_ns(prompt, context, model=pro_model)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.exception("Failed to parse task list JSON: %s", e)
        raise

    task_list = result.get("tasks", [])
    reasoning = result.get("reasoning", "")
    dag_name_value = result.get("dag_name", "generated_dag")
    dag_name = f"{dag_name_value}_{date_str}_{time_str.replace(':', '')}"

    logger.debug("Generated task list: %s", json.dumps(task_list, indent=2))
    logger.info("Task list reasoning: %s", reasoning)

    return task_list, reasoning, dag_name

def choose_functions_for_dag(task_list, available_functions, user_prompt, pipeline_choices, function_choices):
    # Load available pipelines to guide LLM

    #open sample pipeline file to read the task list
    sample_pipeline_location = os.path.join(_TASK_DIR, 'samples','create_IS00_strucutre_and_geothermal_generator.json')
    with open(sample_pipeline_location, 'r', encoding='utf-8') as f:
        sample_pipeline = json.load(f)

    sample_pipeline_location_2 = os.path.join(_TASK_DIR, 'samples','modify_capacities_file_dag_2025-07-24_v2.json')
    with open(sample_pipeline_location_2, 'r', encoding='utf-8') as f:
        sample_pipeline_2 = json.load(f)

    #add variables for date and time
    date_created = datetime.date.today().isoformat()
    time_created = datetime.datetime.now().strftime("%H:%M:%S")
    instruction = f"""
                        Generate a JSON pipeline definition for a new DAG based on these tasks:
                        {task_list}

                        Include keys:
                        - user_input
                        - author
                        - tags
                        - date_created
                        - time_created
                        - retry_policy
                        - default_llm_context
                        - tasks

                        Each task must follow the order above and include:
                        - task_id
                        - task_name
                        - pipeline_name or function_name (only 1 can be chosen)
                        - description
                        - target_level
                        - strategy_action
                        - entity_selection_context
                        - dependencies
                        - on_error

                        The user input is: {user_prompt}

                        The task_id should be incremented starting from 0.

                        For 'pipeline_name', you can choose from one of the existing pipelines below to ensure validity. pipeline are used to interact with the PLEXOS xml database.
                        For most PLEXOS CRUD operation, use a pipeline. Any category, collection, membership, property or attribute interactions should use a pipeline, as they a PLEXOS specific tools:
                        {pipeline_choices}

                        Use functions sparingly, pipelines take priority. For 'function_name', choose from the function registry. Functions are used to interact with other data file e.g. csv, json, excel, etc. This is useful for modifying data files,
                        creating new data files, or reading data files. If a function is called please determine the args and kwargs which will be resolved later. Here are the available functions:
                        {function_choices}

                        For 'strategy_action', the options are:
                        Simple CRUD Actions. The operations are basic create, read, update, delete which are performed on a single model element, Element can be class object, membership, property or attribute. 
                        - create: create a new model element. 
                        - read: read an existing model element.
                        - update: this can only be used to update an object name, memberships or property attributes of an existing model element.
                        - delete: delete an existing model element.

                        Complex Operations. These operations add a modifier to the crud pipeline which split the pipeline into source and destination. This creates a powerful way to perform more complex operations.
                        - clone: create a copy of an existing model element, with a new name and optionally new memberships and property attributes. Intra-class operation
                        - transfer: move a model element from one location to another, updating its context and dependencies. Intra-class or cross-class operation.
                        - merge: combine two or more model elements into a single element, resolving conflicts and dependencies. Intra-class operation.
                        - split: divide a model element into multiple elements, preserving its structure and relationships. Intra-class operation.

                        Be careful with updates, it should be used to update existing model elements. For example if new object is created you cannot update it's properties until the have been created. 
                        If you are cloning a model element the parameter such as object name, memberships and property attributes can be updated as they already exist in the base element.

                        For 'target_level', the options are:
                        - category, object, membership, property, attribute 

                        - 'dependencies' should reference prior tasks by 'task_name' to ensure correct execution order. The dependant classes full output content will be extracted when resolving inputs.

                        The current date is {date_created} and the time is {time_created}.

                        Example DAG format:
                        Example 1: {sample_pipeline}
                        Example 2: {sample_pipeline_2}
                        Use the user scenario '{user_prompt}' to fill in details, focusing strictly on necessary operations, and output valid JSON.
                    """
    
    context = "You are an assistant that creates a PLEXOS pipeline DAG JSON given a scenario." 

    # Invoke LLM
    raw_json = run_open_ai_ns(instruction + '\nScenario: ' + user_prompt, context, model = base_model, reasoning_effort="high")
    # Capture reasoning from DAG generation
    dag_generation_reasoning = raw_json
    # Parse JSON
    dag = json.loads(raw_json)
    logger.debug("DAG Generation Reasoning: %s", dag_generation_reasoning)
    return dag

def load_function_pipeline_index(assistant = None) -> List[Dict[str, str]]:
    """Generate a new DAG JSON via LLM, save it, and update index."""
    pipelines = load_pipeline_index()
    pipeline_choices = '\n'.join(f"- {p['filename']}: {p.get('description','')}" for p in pipelines)
    with open(r'c:\Users\Dante\Documents\AI Architecture\config\function_registry.json', 'r', encoding='utf-8') as f:
        registry = json.load(f)
    # Extract only name / description / args / kwargs for each function.
    # Support BOTH legacy flat list format and new nested category format:
    # Legacy: {"functions": [ {name, description, args, kwargs}, ... ]}
    # New: {"functions": {"CategoryName": {"func_key": {..fn meta..}, ...}, ...}}
    extract: List[Dict[str, Any]] = []
    functions_section = registry.get("functions", {})

    # Case 1: already a list (legacy)
    if isinstance(functions_section, list):
        for fn in functions_section:
            if not isinstance(fn, dict):
                continue
            name = fn.get("name") or fn.get("function_name")
            if not name:
                continue
            extract.append({
                "name": name,
                "description": fn.get("description", ""),
                "args": fn.get("args", {}),
                "kwargs": fn.get("kwargs", {})
            })

    # Case 2: dict of categories
    elif isinstance(functions_section, dict):
        for category, group in functions_section.items():
            # group may be a dict keyed by function name OR a list of dicts
            if isinstance(group, dict):
                for fn_key, fn_val in group.items():
                    if not isinstance(fn_val, dict):
                        continue
                    name = fn_val.get("name") or fn_val.get("function_name") or fn_key
                    extract.append({
                        "name": name,
                        "description": fn_val.get("description", ""),
                        "category": category,
                        "args": fn_val.get("args", {}),
                        "kwargs": fn_val.get("kwargs", {}),
                        "agentic": fn_val.get("agentic", "")  # Default to True if not specified
                    })

            elif isinstance(group, list):
                for fn in group:
                    if not isinstance(fn, dict):
                        continue
                    name = fn.get("name") or fn.get("function_name")
                    if not name:
                        continue
                    extract.append({
                        "name": name,
                        "description": fn.get("description", ""),
                        "category": category,
                        "args": fn.get("args", {}),
                        "kwargs": fn.get("kwargs", {}),
                        "agentic": fn.get("agentic", "")  # Default to True if not specified
                    })
                    
    # (Optional) de-duplicate by name + category
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in extract:
        key = (item.get("category"), item["name"], item['agentic'])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    extract = deduped

    # Filter functions to only include those where 'agentic' is true
    agentic_functions = [
        item for item in extract 
        if str(item.get("agentic", "True")).lower() != 'false'
    ]
    function_choices = json.dumps(agentic_functions, indent=2, ensure_ascii=False)

    return pipeline_choices, function_choices, agentic_functions

def create_new_dag(user_prompt: str, st_tabs=None) -> str:
    pipeline_choices, function_choices, agentic_functions = load_function_pipeline_index()

    if st_tabs:
        with st_tabs[0]:
            with st.spinner("Creating DAG..."):
                st.markdown(
                    f"""
                    <div style="border-left:4px solid #0d6efd;padding:12px 16px;border-radius:6px;background:#f8fbff">
                      <div style="font-size:20px;font-weight:700;color:#0d6efd">⚡ Creating DAG</div>
                      <div style="margin-top:6px;color:#0b2b4a;font-size:14px">For: <strong>{user_prompt}</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("Auto-generated DAG heading — review and adjust as needed.")
                # A lightweight status placeholder that can be updated later during generation steps
                status_placeholder = st.empty()
                status_placeholder.info("Preparing to generate DAG…")

    # Step 1: RAG task list generation
    task_list, task_list_reasoning, dag_name = generate_task_list(user_prompt, pipeline_choices, function_choices)

    dag = choose_functions_for_dag(task_list, agentic_functions, user_prompt, pipeline_choices, function_choices)

    # Write to file    pipeline_id = dag.get('pipeline_id', str(uuid.uuid4()))
    filename = f"{dag_name}.json"
    today = datetime.date.today()
    year = str(today.year)
    month = f"{today.month:02d}"
    day = f"{today.day:02d}"

    filepath = os.path.join("task_lists", year, month, day, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dag, f, indent=2)
    logger.info("Wrote DAG to %s", filepath)

    # Update index
    idx = load_index()
    entry = {
        'pipeline_id': dag.get('pipeline_id'),
        'filename': filename,
        'description': dag.get('description',''),
        'date_created': dag.get('date_created', datetime.date.today().isoformat()),
        'tags': dag.get('tags', [])
    }
    idx.setdefault('dags', []).append(entry)
    save_index(idx)
    logger.info("Updated DAG index with %s", filename)

    # Return filename and combined reasoning
    reasoning = {
        'task_list_reasoning': task_list_reasoning,
    }
    return filepath, reasoning

def choose_or_create_dag(user_prompt: str, st_tabs = None) -> Tuple[str, Dict[str, str]]:
    """Main orchestrator: selects or creates a DAG and returns its filepath plus reasoning."""
    # dags = list_existing_dags()
    # selected = select_existing_dag(user_prompt, dags)
    # if selected:
    #     # If reusing, no new reasoning; just return path
    #     return os.path.join(_TASK_DIR, selected), {'selection': 'reused_existing'}
    # # Create new
    new_filename, reasoning = create_new_dag(user_prompt, st_tabs=st_tabs)
    return os.path.join(new_filename), reasoning

if __name__ == '__main__':
    # Example usage
    prompt = "Modify the H2\Gas Pipeline\Capacities_H2_Year_FID+PCI+PMI.csv file and set all values, which are not 0, to 999. Name the new file Pipeline\Capacities_H2_Year_FID+PCI+PMI_UNLMITED.csv"
    path = choose_or_create_dag(prompt)
    model_description = f"Using DAG: {path[0]}"
    logger.info("Using DAG: %s", path)
