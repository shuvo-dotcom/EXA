# -*- coding: utf-8 -*-
"""
File CRUD Operations Script with AI Integration

This script performs Create, Read, Update, Delete operations on files
with AI assistance for analysis and content generation.

@author: AI Architecture System
"""

import os
import sys
import json
import csv
import pandas as pd
from typing import Union, Dict, Any, Optional
from pathlib import Path

# Add AI integration paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai'))
try:
    from open_ai_calls import run_open_ai_ns
    from openai_code_interpretor_assistant import modify_data_file
except ImportError:
    print("Warning: AI integration modules not found. Some features may be limited.")

class FileCRUD:
    """
    File CRUD operations with AI integration
    """
    
    def __init__(self):
        self.supported_formats = {
            'txt': self._handle_txt,
            'json': self._handle_json,
            'csv': self._handle_csv,
            'xlsx': self._handle_excel,
            'xml': self._handle_xml
        }

    def execute_crud_operation(self, action: str, context: str, user_input: str, input_file_path: str, 
                               output_file_path: str = None, input_data: Union[str, dict, None] = None, 
                               output_structure: str = 'txt', output_extension: str = 'txt') -> Dict[str, Any]:
        """
        Execute CRUD operation on files with AI assistance
        
        Args:
            action: CRUD operation ('create', 'read', 'update', 'delete')
            input_data: Data to process (string, dict, or None)
            input_location: Path to input file
            output_location: Path for output file
            output_structure: Output format ('txt', 'json', 'csv', 'xlsx', 'xml')
            output_extension: File extension for output
            user_request: Specific user request for AI processing
            ai_context: Context for AI calls
            
        Returns:
            Dict with operation results
        """
        
        action = action.lower().strip()
        
        try:
            if action == 'create':
                return self._create_operation(
                                                input_data, output_file_path, output_structure, 
                                                output_extension, user_input, context
                                            )
            elif action == 'read':
                return self._read_operation(
                                                input_file_path, user_input, context
                                            )
            elif action == 'update':
                return self._update_operation(
                                                    input_data, input_file_path, output_file_path,
                                                    output_structure, output_extension, user_input, context
                                                )
            elif action == 'delete':
                return self._delete_operation(input_file_path, user_input, context)
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported action: {action}. Use 'create', 'read', 'update', or 'delete'"
                }
                
        except Exception as e:
            return {
                        "status": "error",
                        "message": f"Error executing {action} operation: {str(e)}"
                    }
    
    def _create_operation(
        self, 
        input_data: Union[str, dict, None], 
        output_location: str, 
        output_structure: str,
        output_extension: str,
        user_request: str,
        ai_context: str
    ) -> Dict[str, Any]:
        """Create a new file with AI assistance"""
        
        if not output_location:
            return {"status": "error", "message": "Output location required for create operation"}
        
        # Ensure output directory exists
        output_path = Path(output_location)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use AI to generate content if needed
        if user_request and not input_data:
            ai_message = f"Generate content for a new {output_structure} file. User request: {user_request}"
            try:
                ai_response = run_open_ai_ns(ai_message, ai_context)
                input_data = ai_response
            except Exception as e:
                return {"status": "error", "message": f"AI content generation failed: {str(e)}"}
        
        # Use code interpreter for complex data operations
        if output_structure in ['csv', 'xlsx'] and user_request:
            try:
                # Create a temporary directory for code interpreter
                temp_dir = output_path.parent
                temp_filename = f"temp_input.{output_extension}"
                output_filename = output_path.name
                
                # Save input data to temp file if provided
                if input_data:
                    temp_path = temp_dir / temp_filename
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        if isinstance(input_data, dict):
                            json.dump(input_data, f, indent=2)
                        else:
                            f.write(str(input_data))
                
                result = modify_data_file(
                    str(temp_dir), 
                    temp_filename if input_data else None,
                    user_request, 
                    output_filename
                )
                
                return {
                    "status": "success",
                    "message": f"File created successfully using code interpreter",
                    "output_path": output_location,
                    "ai_result": result
                }
                
            except Exception as e:
                # Fall back to standard creation if code interpreter fails
                pass
        
        # Standard file creation
        try:
            if output_structure in self.supported_formats:
                self.supported_formats[output_structure](
                    'write', input_data, output_location
                )
            else:
                # Default text file creation
                with open(output_location, 'w', encoding='utf-8') as f:
                    if isinstance(input_data, dict):
                        f.write(json.dumps(input_data, indent=2))
                    else:
                        f.write(str(input_data) if input_data else "")
            
            return {
                "status": "success",
                "message": f"File created successfully",
                "output_path": output_location
            }
            
        except Exception as e:
            return {"status": "error", "message": f"File creation failed: {str(e)}"}
    
    def _read_operation(
        self, 
        input_location: str, 
        user_request: str, 
        ai_context: str
    ) -> Dict[str, Any]:
        """Read and analyze file content with AI"""
        
        if not input_location or not os.path.exists(input_location):
            return {"status": "error", "message": "Input file not found"}
        
        try:
            # Read file content
            file_extension = Path(input_location).suffix.lower().lstrip('.')
            
            if file_extension in self.supported_formats:
                content = self.supported_formats[file_extension]('read', None, input_location)
            else:
                # Default text reading
                with open(input_location, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Use AI to analyze content if user request provided
            ai_analysis = None
            if user_request:
                content_str = ""
                if isinstance(content, pd.DataFrame):
                    content_str = content.to_json(orient='records')
                elif isinstance(content, (dict, list)):
                    content_str = json.dumps(content, indent=2)
                else:
                    content_str = str(content)
                
                ai_message = f"Analyze this file content and respond to: {user_request}\n\nFile content (in JSON format if applicable):\n{content_str}"
                try:
                    ai_analysis = run_open_ai_ns(ai_message, ai_context)
                except Exception as e:
                    ai_analysis = f"AI analysis failed: {str(e)}"
            

            try:
                ai_analysis_json = json.loads(ai_analysis)
            except:
                ai_analysis_json = ai_analysis

            return {
                "status": "success",
                "message": "File read successfully",
                "content": content,
                "ai_analysis": ai_analysis_json,
                "file_path": input_location
            }
            
        except Exception as e:
            return {"status": "error", "message": f"File reading failed: {str(e)}"}
    
    def _update_operation(
        self,
        input_data: Union[str, dict, None],
        input_location: str,
        output_location: str,
        output_structure: str,
        output_extension: str,
        user_request: str,
        ai_context: str
    ) -> Dict[str, Any]:
        """Update file with AI assistance - creates new file instead of overwriting"""
        
        if not input_location or not os.path.exists(input_location):
            return {"status": "error", "message": "Input file not found"}
        
        if not output_location:
            # Generate output location based on input location
            input_path = Path(input_location)
            output_location = str(input_path.parent / f"{input_path.stem}_updated{input_path.suffix}")
        
        try:
            # For complex data updates, use code interpreter
            if output_structure in ['csv', 'xlsx'] and user_request:
                input_path = Path(input_location)
                output_path = Path(output_location)
                
                result = modify_data_file(
                    str(input_path.parent),
                    input_path.name,
                    user_request,
                    output_path.name
                )
                
                return {
                    "status": "success",
                    "message": "File updated successfully using code interpreter",
                    "output_path": output_location,
                    "ai_result": result
                }
            
            # Standard update operation
            # Read existing content
            read_result = self._read_operation(input_location, None, ai_context)
            if read_result["status"] == "error":
                return read_result
            
            existing_content = read_result["content"]
            
            # Use AI to determine update strategy
            if user_request:
                ai_message = f"""Update the following content based on this request: {user_request}
                
                Current content:
                {str(existing_content)[:3000]}...
                
                Additional data to incorporate:
                {str(input_data) if input_data else 'None'}
                
                Provide the updated content."""
                
                try:
                    updated_content = run_open_ai_ns(ai_message, ai_context)
                except Exception as e:
                    return {"status": "error", "message": f"AI update generation failed: {str(e)}"}
            else:
                # Simple append or replace
                if input_data:
                    updated_content = str(existing_content) + "\n" + str(input_data)
                else:
                    updated_content = existing_content
            
            # Create updated file
            create_result = self._create_operation(
                updated_content, output_location, output_structure,
                output_extension, None, ai_context
            )
            
            if create_result["status"] == "success":
                create_result["message"] = "File updated successfully (new file created)"
            
            return create_result
            
        except Exception as e:
            return {"status": "error", "message": f"File update failed: {str(e)}"}
    
    def _delete_operation(self, input_location: str, user_request: str) -> Dict[str, Any]:
        """Delete file operation with safety checks"""
        
        if not input_location or not os.path.exists(input_location):
            return {"status": "error", "message": "Input file not found"}
        
        # Safety check - require explicit confirmation for deletion
        if not user_request or "delete" not in user_request.lower():
            return {
                "status": "warning",
                "message": "Deletion not performed. Please explicitly request deletion in user_request parameter."
            }
        
        try:
            os.remove(input_location)
            return {
                "status": "success",
                "message": f"File deleted successfully",
                "deleted_path": input_location
            }
        except Exception as e:
            return {"status": "error", "message": f"File deletion failed: {str(e)}"}
    
    def _handle_txt(self, operation: str, data: Any, file_path: str) -> Any:
        """Handle text file operations"""
        if operation == 'read':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif operation == 'write':
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(data) if data else "")
    
    def _handle_json(self, operation: str, data: Any, file_path: str) -> Any:
        """Handle JSON file operations"""
        if operation == 'read':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif operation == 'write':
            with open(file_path, 'w', encoding='utf-8') as f:
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except:
                        data = {"content": data}
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _handle_csv(self, operation: str, data: Any, file_path: str) -> Any:
        """Handle CSV file operations"""
        if operation == 'read':
            return pd.read_csv(file_path)
        elif operation == 'write':
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
            elif isinstance(data, str):
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    f.write(data)
            else:
                # Convert to DataFrame if possible
                df = pd.DataFrame([data] if not isinstance(data, list) else data)
                df.to_csv(file_path, index=False)
    
    def _handle_excel(self, operation: str, data: Any, file_path: str) -> Any:
        """Handle Excel file operations"""
        if operation == 'read':
            return pd.read_excel(file_path)
        elif operation == 'write':
            if isinstance(data, pd.DataFrame):
                data.to_excel(file_path, index=False)
            else:
                # Convert to DataFrame
                df = pd.DataFrame([data] if not isinstance(data, list) else data)
                df.to_excel(file_path, index=False)
    
    def _handle_xml(self, operation: str, data: Any, file_path: str) -> Any:
        """Handle XML file operations"""
        if operation == 'read':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif operation == 'write':
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(data) if data else "")


def perform_file_crud(action, user_input, input_file_path, context, output_file_path=None):
    """
    Example usage and testing function
    """
    crud = FileCRUD()
    result = crud.execute_crud_operation(action, context,  user_input, input_file_path, output_file_path=output_file_path)
    print(f"Result: {result}")
    return result['ai_analysis'] if 'ai_analysis' in result else result

    # Example 1: Create a new file with AI-generated content
    # print("Example 1: Creating a new file with AI assistance")
    # result = crud.execute_crud_operation(
    #     action="create",
    #     output_location="output/test_ai_generated.txt",
    #     output_structure="txt",
    #     output_extension="txt",
    #     user_request="Create a sample configuration file for a web application",
    #     ai_context="You are a helpful configuration file generator."
    # )
    # print(f"Create result: {result}")
    
    # # Example 2: Read and analyze an existing file
    # print("\nExample 2: Reading and analyzing a file")
    # result = crud.execute_crud_operation(
    #     action="read",
    #     input_location="README.md",
    #     user_request="Summarize the main purpose and key features of this project",
    #     ai_context="You are a helpful documentation analyzer."
    # )
    # print(f"Read result: {result}")
    
    # # Example 3: Update a file
    # print("\nExample 3: Updating a file")
    # result = crud.execute_crud_operation(
    #     action="update",
    #     input_data="Additional content to add",
    #     input_location="README.md",
    #     output_location="output/README_updated.md",
    #     user_request="Add a section about installation requirements",
    #     ai_context="You are a helpful documentation editor."
    # )
    # print(f"Update result: {result}")
    #do a delete example deleting this file: output/test_ai_generated.txt
    # result = crud.execute_crud_operation(
    #     action="delete",
    #     input_location="output/test_ai_generated.txt",
    #     user_request="Please delete this file permanently",
    #     ai_context="You are a file management assistant."
    # )


if __name__ == "__main__":
    context = """
                    You are being called as part of a system which does  AI research. The system can call agents as internet research, RAG, file search etc.
                    The user wants to know about the import locations for petroleum products such as petrol, kerosene, diesel, and methanol in Europe, then convert these locations
                    into nodes based on the users modelling topology.

                    Task 1: 
                    Search query: where are the import location for petroleum products such as petrol, kerosene, diesel, methonol, in europe. 
                    Response:
                    'Europe\'s seaborne import terminals play a crucial role in supplying the continent with kerosene, jet fuel, and aviation fuel, 
                    with a growing focus on sustainable aviation fuels (SAF), including methanol-based options. While specific "methanol capacity" at 
                    import terminals for direct aviation fuel blending is not explicitly detailed, information on overall fuel import capacities, major ports, 
                    and the burgeoning methanol-to-jet (MtJ) SAF production landscape is available.
                    \n\n**Kerosene/Jet Fuel/Aviation Fuel Import Terminals and Capacity:**\n\nEurope has substantial infrastructure for importing and storing aviation fuels. 
                    Members of FETSA (the Federation of European Tank Storage Associations) operate 768 terminals across Europe, with a collective storage capacity of 125.7 million m³ 
                    and an annual throughput of 1 billion tons of liquid bulk. Many of these strategic terminals are designated as Critical National Infrastructure due to their importance 
                    in supplying energy to industrial, transport, and defense markets, and they also hold strategic reserves for emergencies.\n\nKey import hubs and regions include:\n*   
                    **Dutch ports:** These ports possess significant kerosene storage capacity, supporting major aviation hubs like Amsterdam Schiphol.\n*   **France:** Le Havre and 
                    Marseille are identified as two of the most important hubs, where imported products are delivered to consumption centers via pipelines. The French association within 
                    FETSA represents a total capacity of 28.6 million m³.\n*   **United Kingdom:** The UK is a net importer of aviation fuels, with numerous terminals considered critical 
                    national infrastructure.\n*   **Germany:** Companies represented by Germany\'s UTV manage a total of 12.6 million m³ of liquid bulk storage. The Wilhelmshaven-Hamburg-Rostock 
                    range is particularly significant for imports of oil products and for storage.\n*   **Spain:** Spain has a well-developed and interconnected pipeline network that links 
                    refineries, ports, and industrial facilities. Spain unloaded 0.25 million metric tons of jet fuel in May 2025 in the Mediterranean region.\n*   **Port of Koper, Slovenia:** 
                    This port handles liquid cargo, including jet and diesel fuel imports.\n*   **Rotterdam:** Recognized as the largest European bunker port, indicating substantial fuel handling 
                    capacity.\n\nEuropean jet fuel imports demonstrate significant volumes. Northwest Europe (NWE) imported almost 1.8 million metric tons of jet fuel in May 2025, an increase from 
                    1.2 million metric tons in April and the highest monthly volume since October 2024. The Mediterranean region also saw a surge, with jet imports reaching 0.63 million metric 
                    tons in May. Major sources of these imports include the Middle East, Asia Pacific, UAE, Kuwait, India, South Korea, and Saudi Arabia.
                    \n\n**Methanol Capacity (for Aviation Fuel):**\n\nWhile methanol is discussed as a crucial component for Sustainable Aviation Fuels (SAF), the current focus is on its 
                    *production* for conversion into jet fuel rather than its direct import as a finished aviation fuel or having dedicated methanol import terminals for traditional jet fuel 
                    blending. Europe is a leader in the "methanol-to-jet" (MtJ) pathway for SAF production, hosting over 80% of such projects globally. By 2030, the European SAF project pipeline
                      is projected to include 1.4 million metric tons (Mt) of methanol-to-jet capacity.\n\nE-methanol, produced using renewable energy and biogenic CO₂, is highlighted as a green 
                      alternative that can be easily transported due to its liquid state at room temperature, allowing it to utilize existing liquid fuel infrastructure in harbors. 
                      In 2020, European methanol production capacity was 3.7 Mt, but demand was considerably higher at 7.5 Mt, leading to substantial imports of general methanol, for instance, 
                      from Trinidad. This suggests existing import infrastructure for methanol, which could potentially be leveraged for e-methanol as a SAF feedstock.
                      \n\n**Seasonal Capacity:**\n\nJet fuel consumption in Europe exhibits clear seasonal patterns, surging during the summer months due to increased holiday travel. 
                      Monthly EU and UK jet fuel demand peaked at 5.8 million metric tons in July of last year, indicating a heightened need for import and storage capacity during this 
                      period.\n\n**City/Port Overview:**\n\nIn summary, key European cities and ports involved in seaborne aviation fuel imports and storage include:\n*   
                      **Dutch Ports (e.g., Amsterdam/Rotterdam):** Substantial kerosene storage capacity; Rotterdam is Europe\'s largest bunker port.\n*   **Le Havre, France**\n*   
                      **Marseille, France**\n*   **Wilhelmshaven-Hamburg-Rostock range, Germany**\n*   **Ports in Spain** (e.g., those handling Mediterranean imports)\n*   
                      **Port of Koper, Slovenia**\n*   **London Heathrow and Gatwick (UK):** While airports, they are major consumers, relying on terminal infrastructure for supply.
                      \n\nThe precise Mtpa import terminal capacities specifically for jet fuel at each of these individual ports are not readily available in the provided information, 
                      as capacities are often aggregated by national associations or for liquid bulk in general. Similarly, dedicated import terminal capacity in Mtpa specifically for 
                      methanol used as aviation fuel is not distinctly itemized, given that MtJ is an emerging SAF production pathway.



            """
    user_input = """
                    Read the file obtained and provide a list of 'Joule node' names where the import objects should be located
                    Please return a JSON array of node names.
                    {{
                    'node_names': [<node_name_1>, <node_name_2>, ...], 
                    'reasoning': <reasoning>
                    }}
                    """
    action = "read"
    input_location = r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Sectoral Model\Nodes and Grid\NUTS Regions\NUTS2 Ehighway Mapping.csv'
    # perform_file_crud(user_input, context, action, input_file_path=input_location)
    perform_file_crud(action, user_input, input_file_path=input_location, context=context, output_file_path=None)
