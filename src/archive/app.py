import traceback
import streamlit as st
from components.file_selector import select_plexos_file
import time
from src.pipeline.archive.pipeline_executor_v04 import PipelineExecutor
import src.EMIL.plexos.plexos_extraction_functions_agents as pef
import src.EMIL.plexos.routing_system as rs
from src.ai.PLEXOS.modelling_system_functions import choose_plexos_xml_file
from src.ai.architecture.choose_or_create_dag import choose_or_create_dag
from src.ai.llm_calls.open_ai_calls import run_open_ai_ns as roains

def run_pipeline(executor, user_prompt, pipeline_file_path, function_registry_path):
    if executor.plexos_db_choices['load_model']:
        plexos_model_location = choose_plexos_xml_file(user_prompt)
        if executor.plexos_db_choices['copy_db']:
            print(f"Copying PLEXOS model to a new file: {plexos_model_location}")
            new_suffix = executor.plexos_db_choices.get("new_model_suffix", "_copy")
            plexos_model_location2 = plexos_model_location.replace('.xml', new_suffix + '.xml')
            pef.load_plexos_xml(file_name=plexos_model_location, updated_name=plexos_model_location2, new_copy=True)
        else:
            plexos_model_location2 = plexos_model_location
        executor.plexos_model_location = plexos_model_location2
    else:
        executor.plexos_model_location = None

    return executor.loop_dags(pipeline_file_path, function_registry_path)

def main(user_prompt, test_mode = True, test_dag = None):
    st.title("PLEXOS Pipeline Executor")

    copy_db = None
    if user_prompt:
        model_results = rs.file_copy_option(user_prompt)
        st.write(f"LLM decision: {'Copy the database' if model_results.get('copy_db', False) else 'Do not copy the database'}")

    # Only proceed if user has entered input and LLM has made a decision
    start_time = time.time()

    if test_mode:
        pipeline_file_paths = [test_dag]
    else:
        pipeline_file_paths = choose_or_create_dag(user_prompt)

    attempts = 1

    function_registry_path = r'config\function_registry.json'
    executor = PipelineExecutor(function_registry_path=function_registry_path)
    executor.plexos_db_choices = model_results

    final_outputs = {}
    for pipeline_file_path in pipeline_file_paths:
        for attempt in range(attempts):
            try:
                final_outputs[pipeline_file_path] = run_pipeline(executor, user_prompt, pipeline_file_path, function_registry_path)
                
                break  # If successful, break the loop
            except Exception as e:
                traceback.print_exc()
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < 2:  # If it's not the last attempt
                    st.write("Retrying after a short delay...")
                    time.sleep(5)  # Wait for 5 seconds before retrying
                else:
                    st.error(f"All 3 attempts failed for pipeline: {pipeline_file_path}")
                    # Optionally, re-raise the exception or handle the failure gracefully
                    raise

    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write(f"\nPipeline executed in {elapsed_time:.2f} seconds.")

if __name__ == '__main__':
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

    # user_prompt = """   Modify the TJ Sectoral Model. 
    #                     Add Nuclear SMR Generators in Germany DE01, DE03, and in Austria AT01. Add the properties and use a max capacity of 1000 MW. 
    #                     I'm testing my DAGs ability to process a list of objects, so the addition of the generators in a DAG task.
    #                  """

    # user_prompt = """   Modify the TJ Sectoral Model. 
    #                     I am trying to model the nuclear fusion process.
    #                     Create a copy of the model using the suffix nuclear_fusion.
    #                     Add a new Nuclear Fusion Node object in the category Nuclear Fusion. Add the node FR01_NF.
    #                     Add a new Tritium Breeding Node object in the category Tritium Breeding. Add the node FR01_LB.
    #                     Add a Line object between the Nuclear Fusion Node FR01_NF and the Tritium Breeding Node FR01_LB.
    #                     Add a Generation object in the category Nuclear Fusion. Add the object FR01_NF_GEN.
    #                     Add a Power2X object in the category Tritium Breeding. Add the object FR01_LB_P2X.
    #                     Add a Fuel object called FR01_LB_FUEL in the category Tritium Breeding.
    #                     Connect the fuel to the generator FR01_NF_GEN.
    #                  """

    user_prompt = """   Modify the TJ Sectoral Model. 
                        Add 2 new Gas Nodes 1 called 'e-kerosene import' and the other 'e-methanol import'.    
                        Add 2 new Gas Fields 1 called 'e-kerosene import' and the other 'e-methanol import'.
                        Perform a web search/LLM call to find import terminals for kerosene and methanol into Europe by ship and yearly capacity including 
                        any information on seasonal capacities. 
                        Ensure we have the city name so we can link them to the hydrogen nodes by locating and reading the NUTS regions e-highway file in the 
                        nodes and lines folder.
                        Create pipelines to link the kerosene import nodes to the relevant hydrogen node. Make the capacity of the pipelines match the import 
                        terminal capacities.
                    """

    # user_prompt = """
    #                     Please create hourly Electricity demand profiles for the TYNDP 2026 Scenarios. Use the 'NT', 'NT_HE' and 'NT_LE' Scenarios.
    #                     Use the year 2030, 2035, 2040 and 2050. Split the nodes. Convert to tst format. 
    #                     No need to access any PLEXOS database for now
    #                 """
    test_mode = True
    test_dag = r'task_lists/2025/08/31/tj_sectoral_model_import_nodes_and_kerosene_pipeline_dag_2025-08-31_005114.json'

    main(user_prompt, test_mode=test_mode, test_dag=test_dag)
