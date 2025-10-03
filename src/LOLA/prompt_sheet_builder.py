"""
Generative Prompt Sheet Builder for LLM Report Generation

This module generates hierarchical prompt sheets using LLMs to create structured report outlines.
The structure follows: Aims -> Objectives -> Tasks -> Sub Tasks
Each level is generated sequentially, expanding from the user's input and context.

The system builds from bottom up:
1. Generate Aims based on user request
2. Break each Aim into Objectives (using LLM)
3. Break each Objective into Tasks (max 6 per objective)
4. Break each Task into Sub Tasks
5. Generate External Search configurations for relevant Sub Tasks
6. Generate Text Guidelines for each Sub Task based on context/RAG
7. (Placeholder) Generate Default Charts
"""

import os
import sys
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ai.llm_calls.open_ai_calls import run_open_ai_ns as roains


class PromptSheetBuilder:
    """
    Generates hierarchical prompt sheets for report generation using LLMs.
    
    The prompt sheet structure is generated dynamically:
    - Aims: Top-level goals (generated from user request)
    - Objectives: Specific objectives under each aim (LLM expands)
    - Tasks: Detailed tasks under each objective (LLM expands, max 6)
    - Sub Tasks: Granular sub-tasks under each task (LLM expands)
    - External Search: Search configurations for data retrieval
    - Text Guidelines: Context-specific guidelines for each sub-task
    """
    
    def __init__(self, base_model: str = "gpt-4", context_source: Optional[str] = None, 
                 dag_info: Optional[Dict[str, Any]] = None):
        """
        Initialize the Prompt Sheet Builder.
        
        Args:
            base_model: The LLM model to use for generation
            context_source: Path to context documents or RAG database
            dag_info: Information about available DAG for data extraction
        """
        self.base_model = base_model
        self.context_source = context_source
        self.dag_info = dag_info
        self.prompt_sheet = {
            'aims': [],
            'external_search': [],
            'text_guidelines': [],
            'default_charts': []
        }
    
    def generate_aims(self, user_request: str, context: str = "") -> List[Dict[str, Any]]:
        """
        Generate the top-level Aims based on user request.
        
        Args:
            user_request: User's description of the report they want
            context: Additional context about the project/model
            
        Returns:
            List of aim dictionaries
        """
        message = f"""
                        You are helping to structure a comprehensive energy modeling report.

                        User Request: {user_request}

                        Based on this request, generate 1-3 high-level AIMS for the report.

                        Return your response as a JSON array with this exact structure:
                        [
                            {{
                                "id": 1,
                                "title": "Brief title of the aim",
                                "description": "Detailed description of what this aim encompasses"
                            }}
                        ]

                        Only return the JSON array, no other text.
                    """
        
        try:
            response = roains(message, context, model=self.base_model)
            
            # Parse JSON response
            aims = json.loads(response)
            
            # Validate structure
            if not isinstance(aims, list):
                raise ValueError("Response must be a JSON array")
            
            print(f"✓ Generated {len(aims)} aim(s)")
            return aims
            
        except json.JSONDecodeError as e:
            print(f"Error parsing aims JSON: {e}")
            print(f"Response was: {response}")
            return []
        except Exception as e:
            print(f"Error generating aims: {e}")
            return []
    
    def generate_objectives(self, aim: Dict[str, Any], context: str = "") -> List[Dict[str, Any]]:
        """
        Generate Objectives for a specific Aim.
        
        Args:
            aim: The aim dictionary
            context: Additional context
            
        Returns:
            List of objective dictionaries
        """
        message = f"""
                        You are breaking down a report aim into specific objectives.

                        Aim: {aim['title']}
                        Description: {aim['description']}

                        Generate 3-7 specific OBJECTIVES that together fulfill this aim.
                        Each objective should be a distinct section of the report.

                        Return your response as a JSON array with this exact structure:
                        [
                            {{
                                "id": 1,
                                "aim_id": {aim['id']},
                                "title": "Brief title of the objective",
                                "description": "Detailed description of what this objective covers"
                            }}
                        ]

                        Only return the JSON array, no other text.
                    """
        
        try:
            response = roains(message, context, model=self.base_model)
            objectives = json.loads(response)
            
            print(f"  ✓ Generated {len(objectives)} objective(s) for aim {aim['id']}")
            return objectives
            
        except json.JSONDecodeError as e:
            print(f"Error parsing objectives JSON: {e}")
            return []
        except Exception as e:
            print(f"Error generating objectives: {e}")
            return []
    
    def generate_tasks(self, objective: Dict[str, Any], context: str = "") -> List[Dict[str, Any]]:
        """
        Generate Tasks for a specific Objective (max 6 tasks).
        
        Args:
            objective: The objective dictionary
            context: Additional context
            
        Returns:
            List of task dictionaries
        """
        message = f"""
                        You are breaking down a report objective into specific tasks.

                        Objective: {objective['title']}
                        Description: {objective['description']}

                        Generate 2-6 specific TASKS that together accomplish this objective.
                        Each task should be a concrete action or analysis to perform.

                        Return your response as a JSON array with this exact structure:
                        [
                            {{
                                "id": "1.1",
                                "task_id": 1,
                                "objective_id": {objective['id']},
                                "title": "Brief title of the task",
                                "description": "Detailed description of what this task involves"
                            }}
                        ]

                        Only return the JSON array, no other text.
                        IMPORTANT: Generate maximum 6 tasks.
                    """
        
        try:
            response = roains(message, context, model=self.base_model)
            tasks = json.loads(response)
            
            # Enforce max 6 tasks
            if len(tasks) > 6:
                tasks = tasks[:6]
            
            print(f"    ✓ Generated {len(tasks)} task(s) for objective {objective['id']}")
            return tasks
            
        except json.JSONDecodeError as e:
            print(f"Error parsing tasks JSON: {e}")
            return []
        except Exception as e:
            print(f"Error generating tasks: {e}")
            return []
    
    def generate_sub_tasks(self, task: Dict[str, Any], context: str = "") -> List[Dict[str, Any]]:
        """
        Generate Sub Tasks for a specific Task.
        
        Args:
            task: The task dictionary
            context: Additional context
            
        Returns:
            List of sub-task dictionaries
        """
        message = f"""
                        You are breaking down a report task into specific sub-tasks.

                        Task: {task['title']}
                        Description: {task['description']}

                        Generate 1-5 specific SUB-TASKS that together complete this task.
                        Each sub-task should be a granular, actionable item.

                        Return your response as a JSON array with this exact structure:
                        [
                            {{
                                "id": "{task['id']}.1",
                                "sub_task_id": 1,
                                "task_section_id": "{task['id']}",
                                "task_header": "{task['title']}",
                                "title": "Brief title of the sub-task",
                                "description": "Detailed description of what this sub-task involves",
                                "input": "What input is needed (e.g., 'Text Guidelines', 'External Search', 'PLEXOS Data')",
                                "geographic_level": "Geographic scope (e.g., 'EU', 'National', 'Regional')"
                            }}
                        ]

                        Only return the JSON array, no other text.
                    """
        
        try:
            response = roains(message, context, model=self.base_model)
            sub_tasks = json.loads(response)
            
            print(f"      ✓ Generated {len(sub_tasks)} sub-task(s) for task {task['id']}")
            return sub_tasks
            
        except json.JSONDecodeError as e:
            print(f"Error parsing sub-tasks JSON: {e}")
            return []
        except Exception as e:
            print(f"Error generating sub-tasks: {e}")
            return []
    
    def generate_external_search(self, sub_task: Dict[str, Any], 
                                  context: str = "") -> List[Dict[str, Any]]:
        """
        Generate External Search configurations for a sub-task.
        
        This determines if the sub-task needs internet search or DAG search.
        
        Args:
            sub_task: The sub-task dictionary
            context: Additional context
            
        Returns:
            List of external search configurations (can be empty)
        """
        dag_context = ""
        if self.dag_info:
            dag_context = f"""
            
                                Available DAG Information:
                                {json.dumps(self.dag_info, indent=2)}

                                If this sub-task can benefit from DAG data extraction, include a DAG search.
                                """
                                        
            message = f"""
                                You are determining if a sub-task needs external data sources.

                                Sub-Task: {sub_task['title']}
                                Description: {sub_task['description']}
                                {dag_context}

                                Determine if this sub-task needs:
                                1. Internet search (for current events, policies, news)
                                2. DAG search (for structured data from the energy model)
                                3. No external search

                                If external search is needed, generate search configurations.
                                If not needed, return an empty array.

                                Return your response as a JSON array with this structure:
                                [
                                    {{
                                        "unique_id": 1,
                                        "id": "{sub_task['id']}",
                                        "title": "{sub_task['task_header']}",
                                        "description": "What information to search for",
                                        "prompt": "Specific search query or prompt",
                                        "data_source": "Internet" or "DAG",
                                        "level": "Detail level needed",
                                        "additional_information": "Any additional context"
                                    }}
                                ]

                                If no external search is needed, return: []

                                Only return the JSON array, no other text.
                            """
        
        try:
            response = roains(message, context, model=self.base_model)
            searches = json.loads(response)
            
            if searches:
                print(f"        ✓ Generated {len(searches)} external search(es) for sub-task {sub_task['id']}")
            
            return searches
            
        except json.JSONDecodeError as e:
            print(f"Error parsing external search JSON: {e}")
            return []
        except Exception as e:
            print(f"Error generating external search: {e}")
            return []
    
    def generate_text_guidelines(self, sub_task: Dict[str, Any], 
                                  context: str = "", 
                                  rag_context: str = "") -> Dict[str, Any]:
        """
        Generate Text Guidelines for a sub-task based on context/RAG.
        
        Args:
            sub_task: The sub-task dictionary
            context: General context
            rag_context: Context from RAG/vector database
            
        Returns:
            Text guideline dictionary
        """
        message = f"""
                        You are creating writing guidelines for a specific section of a report.

                        Sub-Task: {sub_task['title']}
                        Description: {sub_task['description']}

                        Context from knowledge base:
                        {rag_context if rag_context else "No specific context available"}

                        General context:
                        {context}

                        Generate text guidelines that will help an AI write this section effectively.
                        Include:
                        - Standard approach: Basic points to cover
                        - Advanced approach: Deeper analysis points
                        - Research topics: Areas to investigate
                        - Research notes: Specific information to include

                        Return your response as a JSON object with this structure:
                        {{
                            "id": "{sub_task['id']}",
                            "description": "What this section should accomplish",
                            "standard": "Standard level content guidance",
                            "advanced": "Advanced level content guidance",
                            "research_topics": "Topics to research",
                            "research_notes": "Specific notes and information to include"
                        }}

                        Only return the JSON object, no other text.
                    """
        
        try:
            response = roains(message, context, model=self.base_model)
            guideline = json.loads(response)
            
            print(f"        ✓ Generated text guidelines for sub-task {sub_task['id']}")
            return guideline
            
        except json.JSONDecodeError as e:
            print(f"Error parsing text guidelines JSON: {e}")
            return {}
        except Exception as e:
            print(f"Error generating text guidelines: {e}")
            return {}
    
    def generate_default_charts(self, prompt_sheet: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Placeholder function for generating default chart configurations.
        
        This will be implemented later to explore PLEXOS solution files
        and determine what can be extracted from the model.
        
        Args:
            prompt_sheet: The complete prompt sheet structure
            
        Returns:
            List of chart configurations (currently empty)
        """
        print("\n📊 Default Charts generation is a placeholder (to be implemented)")
        print("    This will require PLEXOS solution file exploration")
        
        # TODO: Implement PLEXOS solution file analysis
        # - Explore available properties
        # - Determine available data series
        # - Match to report sections
        # - Generate chart configurations
        
        return []
    
    def build_complete_prompt_sheet(self, user_request: str, 
                                     context: str = "",
                                     rag_context: str = "") -> Dict[str, Any]:
        """
        Build the complete prompt sheet from user request.
        
        This generates the entire hierarchical structure:
        1. Aims (from user request)
        2. Objectives (for each aim)
        3. Tasks (for each objective, max 6)
        4. Sub Tasks (for each task)
        5. External Search (for relevant sub-tasks)
        6. Text Guidelines (for each sub-task)
        7. Default Charts (placeholder)
        
        Args:
            user_request: User's description of the report
            context: General context about the project
            rag_context: Context from RAG/vector database
            
        Returns:
            Complete prompt sheet structure
        """
        print("\n" + "="*70)
        print("BUILDING PROMPT SHEET")
        print("="*70)
        print(f"\nUser Request: {user_request}\n")
        
        # Step 1: Generate Aims
        print("\n[1/7] Generating Aims...")
        aims = self.generate_aims(user_request, context)
        
        if not aims:
            print("❌ Failed to generate aims. Aborting.")
            return self.prompt_sheet
        
        # Process each aim
        for aim in aims:
            print(f"\n{'='*70}")
            print(f"Processing Aim {aim['id']}: {aim['title']}")
            print(f"{'='*70}")
            
            # Step 2: Generate Objectives
            print(f"\n[2/7] Generating Objectives for Aim {aim['id']}...")
            objectives = self.generate_objectives(aim, context)
            aim['objectives'] = objectives
            
            # Process each objective
            for objective in objectives:
                print(f"\n  Processing Objective {objective['id']}: {objective['title']}")
                
                # Step 3: Generate Tasks (max 6)
                print(f"\n  [3/7] Generating Tasks for Objective {objective['id']}...")
                tasks = self.generate_tasks(objective, context)
                objective['tasks'] = tasks
                
                # Process each task
                for task in tasks:
                    print(f"\n    Processing Task {task['id']}: {task['title']}")
                    
                    # Step 4: Generate Sub Tasks
                    print(f"\n    [4/7] Generating Sub Tasks for Task {task['id']}...")
                    sub_tasks = self.generate_sub_tasks(task, context)
                    task['sub_tasks'] = sub_tasks
                    
                    # Process each sub-task
                    for sub_task in sub_tasks:
                        
                        # Step 5: Generate External Search
                        print(f"\n      [5/7] Generating External Search for Sub Task {sub_task['id']}...")
                        searches = self.generate_external_search(sub_task, context)
                        if searches:
                            self.prompt_sheet['external_search'].extend(searches)
                        
                        # Step 6: Generate Text Guidelines
                        print(f"\n      [6/7] Generating Text Guidelines for Sub Task {sub_task['id']}...")
                        guideline = self.generate_text_guidelines(sub_task, context, rag_context)
                        if guideline:
                            self.prompt_sheet['text_guidelines'].append(guideline)
            
            self.prompt_sheet['aims'].append(aim)
        
        # Step 7: Generate Default Charts (placeholder)
        print(f"\n[7/7] Generating Default Charts...")
        charts = self.generate_default_charts(self.prompt_sheet)
        self.prompt_sheet['default_charts'] = charts
        
        print("\n" + "="*70)
        print("✓ PROMPT SHEET BUILDING COMPLETE")
        print("="*70)
        
        return self.prompt_sheet
    
    def export_to_json(self, output_file: str):
        """
        Export the prompt sheet to a JSON file.
        
        Args:
            output_file: Path to save the JSON file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.prompt_sheet, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Prompt sheet exported to: {output_file}")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the generated prompt sheet.
        
        Returns:
            Dictionary with counts of various elements
        """
        stats = {
            'aims': len(self.prompt_sheet['aims']),
            'objectives': 0,
            'tasks': 0,
            'sub_tasks': 0,
            'external_searches': len(self.prompt_sheet['external_search']),
            'text_guidelines': len(self.prompt_sheet['text_guidelines']),
            'default_charts': len(self.prompt_sheet['default_charts'])
        }
        
        for aim in self.prompt_sheet['aims']:
            stats['objectives'] += len(aim.get('objectives', []))
            for obj in aim.get('objectives', []):
                stats['tasks'] += len(obj.get('tasks', []))
                for task in obj.get('tasks', []):
                    stats['sub_tasks'] += len(task.get('sub_tasks', []))
        
        return stats


def main():
    """Example usage of the Prompt Sheet Builder"""
    
    # Example user request
    user_request = """
    Create a comprehensive report on the Joule multi-carrier energy model results.
    The report should cover the hydrogen sector expansion, electricity system integration,
    demand projections, and policy recommendations for 2030-2050.
    """
    
    # Context about the project
    context = """
    The Joule model is a multi-carrier energy system model covering the EU.
    It models hydrogen, electricity, natural gas, and heat sectors with hourly resolution.
    The model covers the period 2025-2050 with 5-year time steps.
    """
    
    # DAG information (if available)
    dag_info = {
        "name": "PLEXOS Energy Model DAG",
        "description": "Directed Acyclic Graph for extracting data from PLEXOS solution files",
        "available_queries": [
            "Generator capacity by fuel type",
            "Annual generation by region",
            "Transmission flows",
            "System costs"
        ]
    }
    
    # Initialize the builder
    print("Initializing Prompt Sheet Builder...")
    builder = PromptSheetBuilder(
        base_model="gpt-4",
        dag_info=dag_info
    )
    
    # Build the prompt sheet
    prompt_sheet = builder.build_complete_prompt_sheet(
        user_request=user_request,
        context=context,
        rag_context=""  # Would come from vector database
    )
    
    # Export to JSON
    output_file = os.path.join(os.path.dirname(__file__), "generated_prompt_sheet.json")
    builder.export_to_json(output_file)
    
    # Print statistics
    stats = builder.get_statistics()
    print("\n" + "="*70)
    print("PROMPT SHEET STATISTICS")
    print("="*70)
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("\n✓ Prompt Sheet Builder completed successfully")


if __name__ == "__main__":
    main()
