#!/usr/bin/env python3
"""
AI-Powered JSON Explorer
An intelligent JSON/YAML explorer that uses LLM to navigate data structures and find the right
key-value pairs based on natural language user input.
"""


import os
import sys
import json
import yaml
from pathlib import Path
import traceback
from typing import List, Dict, Any, Optional, Tuple, Union

top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

from src.tools.simple_file_browser import SimpleFileBrowser
import src.ai.file_operations.ai_finder_config as config
from src.ai.llm_calls.open_ai_calls import run_open_ai_ns as roains

base_model = 'openai/gpt-oss-120b'

context = """
            You are an Agent running as part of an energy modelling system focusing on mainly Europe model that extends global. 
            You are in charge of exploring JSON/YAML data structures to find the most relevant data paths for a given user query.
            """

class AIJSONExplorer:
    """AI-powered JSON/YAML explorer using LLM for intelligent navigation."""
    
    def __init__(self, api_key_file: str = None):
        """Initialize the AI JSON explorer."""
        self.browser = SimpleFileBrowser()
        self.conversation_history = []
        self.decision_history = []  # Track LLM decisions with full context
        self.visited_paths = set()  # Track visited JSON paths to avoid loops
        self.rejected_data = []     # Track data that was examined but rejected
        self.current_data = None
        self.current_path = []      # Current position in JSON structure
        self.search_context = ""
        self.found_data = []
        self.json_files = []        # Available JSON/YAML files
        
        # Enhanced retry system attributes
        self.global_failed_paths = set()     # Paths that failed across attempts
        self.global_rejected_data = []       # Data rejected across attempts  
        self.global_failed_decisions = []    # Failed decisions across attempts
        self.attempt_number = 1              # Current attempt number
        
        # Conversation history for LLM context
        self.llm_history = []                # History of all LLM interactions
        
        # Track explored files
        self.explored_files = set()          # Files we've already explored
        
        # Loop prevention attributes
        self.back_to_root_count = 0          # Track how many times we've gone back to root
        self.path_visit_count = {}           # Track visits to each path to prevent loops
        
        print(f"🤖 AI JSON Explorer initialized")

    def load_json_file(self, file_path: str) -> Dict[str, Any]:
        """Load JSON or YAML file and return parsed data."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.lower().endswith('.yaml') or file_path.lower().endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            print(f"❌ Error loading file {file_path}: {e}")
            return {}

    def get_available_json_files(self, start_path: str = None) -> List[str]:
        """Get list of JSON/YAML files in the workspace."""
        if start_path is None:
            start_path = os.getcwd()
        
        json_files = []
        for root, dirs, files in os.walk(start_path):
            for file in files:
                if file.lower().endswith(('.yaml', '.yml')): #'.json', 
                    json_files.append(os.path.join(root, file))
        return json_files

    def get_json_structure_summary(self, data: Dict[str, Any], max_depth: int = 2) -> Dict[str, Any]:
        """Get a summary of JSON structure for LLM analysis."""
        def summarize_level(obj, depth=0):
            if depth >= max_depth:
                return "..."
            
            if isinstance(obj, dict):
                summary = {}
                for key, value in list(obj.items())[:10]:  # Limit keys shown
                    if isinstance(value, (dict, list)):
                        summary[key] = summarize_level(value, depth + 1)
                    else:
                        summary[key] = f"<{type(value).__name__}>"
                if len(obj) > 10:
                    summary["..."] = f"and {len(obj) - 10} more keys"
                return summary
            elif isinstance(obj, list):
                if len(obj) == 0:
                    return "[]"
                elif len(obj) == 1:
                    return [summarize_level(obj[0], depth + 1)]
                else:
                    return [summarize_level(obj[0], depth + 1), f"... and {len(obj) - 1} more items"]
            else:
                return f"<{type(obj).__name__}>"
        
        return summarize_level(data)

    def get_current_level_context(self) -> Dict[str, Any]:
        """Get context about current position in JSON structure."""
        if not self.current_data:
            return {"error": "No data loaded"}
        
        # Navigate to current position
        current_level = self.current_data
        for path_part in self.current_path:
            if isinstance(current_level, dict) and path_part in current_level:
                current_level = current_level[path_part]
            elif isinstance(current_level, list) and isinstance(path_part, int) and path_part < len(current_level):
                current_level = current_level[path_part]
            else:
                return {"error": f"Invalid path: {'.'.join(map(str, self.current_path))}"}
        
        # Check size of remaining structure to decide if we should show full content
        remaining_structure_str = str(current_level)
        remaining_size = len(remaining_structure_str)
        
        context = {
            "current_path": ".".join(map(str, self.current_path)) if self.current_path else "root",
            "current_level_type": type(current_level).__name__,
            "remaining_structure_size": remaining_size,
            "can_show_full_structure": remaining_size <= 3500,  # 2000 characters threshold
        }
        
        # If structure is small enough, include full content
        if context["can_show_full_structure"]:
            context["full_remaining_structure"] = current_level
            context["structure_is_small"] = True
        else:
            context["structure_is_small"] = False
            # Show limited preview for large structures
            if isinstance(current_level, dict):
                context["available_keys"] = list(current_level.keys())  # Limit for token efficiency
                context["total_keys"] = len(current_level)
                context["structure_preview"] = self.get_json_structure_summary(current_level, max_depth=1)
            elif isinstance(current_level, list):
                context["list_length"] = len(current_level)
                context["item_types"] = list(set(type(item).__name__ for item in current_level[:10]))
                if current_level:
                    context["sample_items"] = current_level[:3]
            else:
                context["value"] = current_level
                context["value_type"] = type(current_level).__name__

        # Extract any "description" fields that exist at this level so we can
        # pass them to the LLM as a constant helpful hint. We gather descriptions
        # for keys (or list indices) at the current level and truncate long text.
        descriptions_at_level: Dict[str, str] = {}
        try:
            if isinstance(current_level, dict):
                for k, v in current_level.items():
                    desc = None
                    if isinstance(v, dict) and 'description' in v:
                        desc = v.get('description')
                    elif isinstance(v, list):
                        # find first list element that contains a description
                        for item in v:
                            if isinstance(item, dict) and 'description' in item:
                                desc = item.get('description')
                                break
                    if desc:
                        if isinstance(desc, str) and len(desc) > 300:
                            desc = desc[:300] + "... [truncated]"
                        descriptions_at_level[str(k)] = desc
            elif isinstance(current_level, list):
                for idx, item in enumerate(current_level[:50]):
                    if isinstance(item, dict) and 'description' in item:
                        desc = item.get('description')
                        if isinstance(desc, str) and len(desc) > 300:
                            desc = desc[:300] + "... [truncated]"
                        descriptions_at_level[str(idx)] = desc
        except Exception:
            # Be resilient: don't fail the whole traversal because of unexpected types
            descriptions_at_level = {}

        context['descriptions_at_level'] = descriptions_at_level
        if descriptions_at_level:
            # A compact joint summary string helpful for quick LLM context
            try:
                context['descriptions_summary'] = json.dumps(descriptions_at_level, indent=2)
            except Exception:
                context['descriptions_summary'] = str(descriptions_at_level)
        
        return context

    def parse_llm_response(self, response: str, context: str = "unknown") -> Dict[str, Any]:
        """Parse LLM response with better error handling and fallback."""
        if not response or not response.strip():
            print(f"⚠️ Empty response in {context}, using fallback")
            return {
                "action": "back",
                "reasoning": "Empty LLM response, going back to previous level",
                "choice": "back",
                "confidence": 0.1
            }
        
        try:
            # Try to parse as JSON directly
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error in {context}: {e}")
            print(f"Raw response: {response[:500]}")
            
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try to find any JSON-like structure
            curly_match = re.search(r'\{[\s\S]*\}', response)
            if curly_match:
                try:
                    return json.loads(curly_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            # Fallback decision
            print(f"⚠️ Unable to parse response in {context}, using fallback")
            return {
                "action": "back",
                "reasoning": f"Failed to parse LLM response: {str(e)}",
                "choice": "back", 
                "confidence": 0.1
            }

    def make_llm_call(self, prompt: str, context: str = None) -> str:
        """Make a call to the LLM API."""
        if context is None:
            # Use the global context variable
            context = globals().get('context', 'You are a helpful assistant.')
            
        response = roains(prompt, context, history=self.llm_history, model=base_model)
        
        # Append the assistant's response to the history for future context
        self.llm_history.append({"role": "assistant", "content": response})
        
        print(f"🔍 LLM Response (first 200 chars): {str(response)[:200]}")  # Debug output
        if not response or not response.strip():
            print("⚠️ Empty or None response from LLM")
            return self.fallback_decision(prompt)
        return response

    def fallback_decision(self, prompt: str) -> str:
        """Fallback decision making when LLM is not available."""
        return json.dumps({
            "action": "manual",
            "reasoning": "LLM not available. Manual navigation required.",
            "choice": "user_input",
            "confidence": 0.0
        })

    def create_file_selection_prompt(self, user_input: str, json_files: List[str]) -> str:
        """Create prompt for LLM to select which JSON/YAML files to explore."""
        system_prompt = config.SYSTEM_PROMPTS.get("navigation", "You are a helpful assistant.")
        
        # Add information about previously explored files
        explored_files_info = ""
        if hasattr(self, 'explored_files') and self.explored_files:
            explored_files_info = f"\n\nPREVIOUSLY EXPLORED FILES (avoid these):\n" + "\n".join([f"- {Path(f).name}" for f in self.explored_files])
        
        prompt = f"""
                    USER QUERY: "{user_input}"

                    AVAILABLE JSON/YAML FILES:
                    {self.format_file_list(json_files)}{explored_files_info}

                    CONVERSATION HISTORY:
                    {self.format_conversation_history()}

                    TASK: Select the most relevant JSON/YAML files to explore based on the user's query.
                    You can select multiple files that might contain relevant data. If we've already explored some files and didn't find what we need, choose different files that might contain the relevant data.

                    Respond with a JSON object containing:
                    {{
                        "action": "select_files",
                        "reasoning": "Explain why these files are most relevant to the user's query", 
                        "choices": [file_number1, file_number2, ...],
                        "confidence": 0.0-1.0
                    }}

                    Choose the file numbers (1-{len(json_files)}) that seem most relevant to the user's query.
                    Select as many as you think are relevant, but prioritize the most important ones.
                    """
        return prompt

    def create_navigation_prompt(self, user_input: str, context: Dict[str, Any]) -> str:
        """Create prompt for LLM to decide JSON navigation."""
        system_prompt = config.SYSTEM_PROMPTS.get("navigation", "You are a helpful assistant.")
        
        # Check if we can show the full remaining structure
        if context.get("can_show_full_structure", False):
            prompt = f"""
                        USER QUERY: "{user_input}"

                        CURRENT JSON CONTEXT:
                        - Current path: {context.get('current_path', 'root')}
                        - Remaining structure size: {context.get('remaining_structure_size', 0)} characters
                        - Structure is small enough to view completely

                        FULL REMAINING STRUCTURE:
                        {json.dumps(context.get('full_remaining_structure', {}), indent=2)}

                        DECISION HISTORY AND LEARNING:
                        {self.format_conversation_history()}

                        DESCRIPTIONS AT CURRENT LEVEL:
                        {context.get('descriptions_summary', '{}')}

                        TASK: Since the remaining structure is small ({context.get('remaining_structure_size', 0)} chars), 
                        you can see the complete data. Analyze it and determine if it contains what the user is looking for.

                        Respond with a JSON object containing:
                        {{
                            "action": "found|back|complete",
                            "reasoning": "Explain your analysis of the complete data structure", 
                            "choice": "specific_path|back|current_path",
                            "confidence": 0.0-1.0,
                            "extracted_data": "specific data that answers the query, or null if not found"
                        }}

                        Actions:
                        - "found": Found specific data that answers the query (choice = specific path to the data)
                        - "complete": The entire current structure answers the query (choice = current path)
                        - "back": This data doesn't contain what we're looking for (choice = "back")

                        IMPORTANT: Since you can see the full structure, be specific about the exact path to relevant data.
                        """
        else:
            prompt = f"""
                        USER QUERY: "{user_input}"

                        CURRENT JSON CONTEXT:
                        - Current path: {context.get('current_path', 'root')}
                        - Current level type: {context.get('current_level_type', 'Unknown')}
                        - Remaining structure size: {context.get('remaining_structure_size', 0)} characters (too large to show completely)
                        - Available keys: {context.get('available_keys', [])}
                        - Total keys: {context.get('total_keys', 0)}
                        - Structure preview: {context.get('structure_preview', {})}

                        DECISION HISTORY AND LEARNING:
                        {self.format_conversation_history()}

                        DESCRIPTIONS AT CURRENT LEVEL:
                        {context.get('descriptions_summary', 'No descriptions available at this level')}

                        FAILURE PREVENTION:
                        {self.format_failure_prevention_context()}

                        TASK: The remaining structure is large ({context.get('remaining_structure_size', 0)} chars), 
                        so we need to navigate step by step. Analyze the available keys and decide the next step.

                        IMPORTANT: Review the DECISION HISTORY above to avoid repeating failed actions!
                        - If you previously explored a key and it failed, don't try again
                        - If you examined data at this location before, consider going elsewhere
                        - Learn from previous reasoning and outcomes
                        - Prioritize unexplored keys over previously visited ones

                        Respond with a JSON object containing:
                        {{
                            "action": "navigate|examine|back|complete",
                            "reasoning": "Explain your decision process INCLUDING what you learned from history", 
                            "choice": "key_name|current|back|found_path",
                            "confidence": 0.0-1.0
                        }}

                        Actions:
                        - "navigate": Choose a key to explore deeper (choice = exact key name from available_keys)
                        - "examine": Look at current level data in detail (choice = "current")
                        - "back": Go back to parent level (choice = "back") - use when current location seems wrong
                        - "complete": Found what we're looking for (choice = current path as string)

                        IMPORTANT: 
                        - For "navigate": use the EXACT key name from the available_keys list
                        - For "back": always use choice = "back" 
                        - For "examine": always use choice = "current"
                        - AVOID repeating actions that failed before (check DECISION HISTORY)
                        """
        return prompt

    def create_data_analysis_prompt(self, user_input: str, current_data: Any, current_path: str) -> str:
        """Create prompt for LLM to analyze current JSON data."""
        system_prompt = config.SYSTEM_PROMPTS.get("file_analysis", "You are a helpful assistant.")
        
        # Truncate data for display if too large
        data_preview = str(current_data)
        if len(data_preview) > 1000:
            data_preview = data_preview[:1000] + "... [truncated]"
        
        prompt = f"""
                        USER QUERY: "{user_input}"

                        CURRENT DATA AT PATH: {current_path}
                        {data_preview}

                        CONVERSATION HISTORY:
                        {self.format_conversation_history()}

                        DESCRIPTIONS AT CURRENT LEVEL:
                        {json.dumps(self.get_current_level_context().get('descriptions_at_level', {}), indent=2)}

                        TASK: Analyze the current data and determine if it contains what the user is looking for.

                        Respond with a JSON object:
                        {{
                            "action": "found|continue|back",
                            "reasoning": "Explain your analysis of the data",
                            "choice": "current_path|explore_more|back",
                            "confidence": 0.0-1.0,
                            "extracted_data": "relevant data if found, null otherwise"
                        }}

                        Actions:
                        - "found": This data answers the user's query (choice = current path)
                        - "continue": Need to explore more (choice = "explore_more")
                        - "back": This data is not relevant (choice = "back")
                        """
        return prompt

    def format_conversation_history(self) -> str:
        """Format conversation history for LLM context."""
        if not self.conversation_history:
            return "No previous conversation history."
        
        formatted = []
        for i, entry in enumerate(self.conversation_history[-5:]):  # Show last 5 entries
            formatted.append(f"{i+1}. {entry['description']}")
            if 'outcome' in entry:
                formatted.append(f"   Outcome: {entry['outcome']}")
        
        return "\n".join(formatted)

    def format_failure_prevention_context(self) -> str:
        """Format failure prevention context for LLM."""
        context = []
        
        if self.global_failed_paths:
            failed_paths = list(self.global_failed_paths)[:5]  # Show max 5
            context.append(f"❌ Previously failed paths: {failed_paths}")
        
        if self.global_failed_decisions:
            failed_decisions = self.global_failed_decisions[-3:]  # Show last 3
            for decision in failed_decisions:
                context.append(f"❌ Failed decision: {decision.get('action', 'unknown')} -> {decision.get('choice', 'unknown')}")
        
        if not context:
            return "No previous failures to avoid."
        
        return "\n".join(context)

    def log_decision(self, decision_type: str, decision: Dict[str, Any], context: Dict[str, Any]):
        """Log a decision for history tracking."""
        entry = {
            "type": decision_type,
            "decision": decision,
            "context": context,
            "timestamp": self.get_timestamp()
        }
        self.decision_history.append(entry)
        
        # Add to conversation history for LLM context
        description = f"{decision_type}: {decision.get('action', 'unknown')} -> {decision.get('choice', 'unknown')}"
        self.conversation_history.append({
            "description": description,
            "reasoning": decision.get('reasoning', 'No reasoning provided')
        })

    def mark_path_failed(self, path: Union[str, List[str]]):
        """Record a path (dot-string or list) as failed so we avoid re-traversal."""
        try:
            if isinstance(path, list):
                path_str = ".".join(map(str, path)) if path else "root"
            else:
                path_str = str(path) if path else "root"
            self.global_failed_paths.add(path_str)
            self.global_failed_decisions.append({
                "action": "marked_failed",
                "choice": path_str,
                "timestamp": self.get_timestamp()
            })
        except Exception:
            pass

    def get_timestamp(self) -> str:
        """Get current timestamp."""
        import datetime
        return datetime.datetime.now().strftime("%H:%M:%S")

    def format_file_list(self, files: List[str]) -> str:
        """Format file list for display with schema and description information."""
        formatted = []
        for i, file_path in enumerate(files, 1):
            file_name = Path(file_path).name
            relative_path = str(Path(file_path).relative_to(os.getcwd()))
            
            # Try to extract schema and description from the file
            schema_info = ""
            description_info = ""
            try:
                file_data = self.load_json_file(file_path)
                if isinstance(file_data, dict):
                    # Extract schema
                    if 'schema' in file_data:
                        schema = file_data['schema']
                        schema_str = json.dumps(schema, indent=2)
                        if len(schema_str) > 200:
                            schema_str = schema_str[:200] + "... [truncated]"
                        schema_info = f"\n    Schema: {schema_str}"
                    
                    # Extract description
                    if 'description' in file_data:
                        description = file_data['description']
                        if len(description) > 100:
                            description = description[:100] + "... [truncated]"
                        description_info = f"\n    Description: {description}"
            except Exception as e:
                schema_info = f"\n    Schema: Error loading - {str(e)}"
            
            formatted.append(f"{i}. {file_name} ({relative_path}){description_info}{schema_info}")
        return "\n".join(formatted)

    def log_step(self, step_description: str):
        """Log a step in the exploration process."""
        print(f"📝 {step_description}")

    def _explore_json_structure(self, user_input: str, file_path: str) -> Optional[Tuple[str, Any]]:
        """Explore JSON structure to find relevant data."""
        max_steps = config.MAX_SEARCH_STEPS
        step_count = 0
        
        while step_count < max_steps:
            step_count += 1
            print(f"\n🔄 Step {step_count}")
            
            # Check for loop prevention - track current path visits
            current_path_str = ".".join(map(str, self.current_path)) if self.current_path else "root"
            self.path_visit_count[current_path_str] = self.path_visit_count.get(current_path_str, 0) + 1
            
            # If we've visited this path too many times, force a different direction
            if self.path_visit_count[current_path_str] > 3:
                print(f"🔄 Path '{current_path_str}' visited {self.path_visit_count[current_path_str]} times - forcing different direction")
                if self.current_path:
                    popped = self.current_path.pop()
                    # mark the recently-exited child path as failed to avoid re-entering
                    child_path = ".".join(map(str, (self.current_path + [popped]) if self.current_path else [popped]))
                    self.mark_path_failed(child_path)
                    print(f"🔙 Forced back one level to avoid loop (marked failed: {child_path})")
                    continue
            
            # Get current level context
            context = self.get_current_level_context()
            
            if context.get('error'):
                print(f"❌ Error accessing data: {context['error']}")
                return None
            
            print(f"📍 Current path: {context['current_path']}")
            print(f"📊 Level type: {context['current_level_type']}")
            print(f"📏 Remaining structure size: {context.get('remaining_structure_size', 0)} characters")
            
            # Check if we can show the full structure due to small size
            if context.get("can_show_full_structure", False):
                print(f"💡 Structure is small enough - showing complete view to LLM")
                
                # Ask LLM to analyze the complete small structure
                nav_prompt = self.create_navigation_prompt(user_input, context)
                nav_response = self.make_llm_call(nav_prompt)
                
                try:
                    nav_decision = self.parse_llm_response(nav_response, "small_structure_analysis")
                    
                    # Log the decision
                    self.log_decision("small_structure_analysis", nav_decision, context)
                    
                    print(f"🤖 LLM Decision: {nav_decision['action']}")
                    print(f"💭 Reasoning: {nav_decision['reasoning']}")
                    
                    if nav_decision['action'] == 'found':
                        specific_path = nav_decision.get('choice', 'current_path')
                        extracted_data = nav_decision.get('extracted_data')
                        
                        if specific_path and specific_path != 'current_path':
                            print(f"🎉 Found specific data at path: {specific_path}")
                            return (specific_path, extracted_data if extracted_data is not None else self._get_current_data())
                        else:
                            current_path_str = ".".join(map(str, self.current_path)) if self.current_path else "root"
                            print(f"🎉 Found relevant data at current path: {current_path_str}")
                            return (current_path_str, extracted_data if extracted_data is not None else self._get_current_data())
                            
                    elif nav_decision['action'] == 'complete':
                        current_path_str = ".".join(map(str, self.current_path)) if self.current_path else "root"
                        current_data = self._get_current_data()
                        print(f"🎉 Complete structure answers query at: {current_path_str}")
                        return (current_path_str, current_data)
                        
                    elif nav_decision['action'] == 'back':
                        if self.current_path:
                            self.current_path.pop()
                            print("🔙 Going back one level")
                            
                            # Check if we're back at root after going back
                            if not self.current_path:
                                self.back_to_root_count += 1
                                print(f"🏠 Back at root level (count: {self.back_to_root_count})")
                                
                                # If we've gone back to root too many times, switch files
                                if self.back_to_root_count >= 3:
                                    print("🔄 Excessive back-to-root navigation detected - switching to alternative file")
                                    return self._switch_to_alternative_file(user_input, step_count, max_steps)
                        else:
                            print("❌ Already at root level, no relevant data found")
                            self.back_to_root_count += 1
                            if self.back_to_root_count >= 3:
                                print("🔄 Excessive back-to-root navigation detected - switching to alternative file")
                                return self._switch_to_alternative_file(user_input, step_count, max_steps)
                            return None
                            
                except Exception as e:
                    print(f"❌ Error in small structure analysis: {e}")
                    return None
                    
            elif context['current_level_type'] == 'dict':
                print(f"🔑 Available keys: {len(context.get('available_keys', []))}")
                
                # Ask LLM to decide next navigation step
                nav_prompt = self.create_navigation_prompt(user_input, context)
                nav_response = self.make_llm_call(nav_prompt)
                
                try:
                    nav_decision = self.parse_llm_response(nav_response, "json_navigation")
                    
                    # Log the navigation decision
                    self.log_decision("navigation", nav_decision, context)
                    
                    print(f"🤖 LLM Decision: {nav_decision['action']}")
                    print(f"💭 Reasoning: {nav_decision['reasoning']}")
                    
                    if nav_decision['action'] == 'navigate':
                        # Navigate deeper into JSON structure
                        key_choice = nav_decision['choice']
                        if key_choice in context.get('available_keys', []):
                            prospective_path = (self.current_path + [key_choice]) if self.current_path else [key_choice]
                            prospective_path_str = ".".join(map(str, prospective_path))
                            if prospective_path_str in self.global_failed_paths:
                                print(f"⛔ Skipping navigation to previously failed path: {prospective_path_str}")
                            else:
                                self.current_path.append(key_choice)
                                print(f"🚀 Navigating to key: {key_choice}")
                        else:
                            print(f"❌ Invalid key choice: {key_choice}")
                            return None
                            
                    elif nav_decision['action'] == 'examine':
                        # Examine current level data
                        current_data = self._get_current_data()
                        current_path_str = ".".join(map(str, self.current_path)) if self.current_path else "root"
                        
                        analysis_prompt = self.create_data_analysis_prompt(user_input, current_data, current_path_str)
                        analysis_response = self.make_llm_call(analysis_prompt)
                        
                        try:
                            analysis_decision = self.parse_llm_response(analysis_response, "data_analysis")
                            print(f"📊 Analysis: {analysis_decision['reasoning']}")
                            
                            if analysis_decision['action'] == 'found':
                                print(f"🎉 Found relevant data at path: {current_path_str}")
                                return (current_path_str, analysis_decision.get('extracted_data', current_data))
                            elif analysis_decision['action'] == 'back':
                                if self.current_path:
                                    popped = self.current_path.pop()
                                    child_path = ".".join(map(str, (self.current_path + [popped]) if self.current_path else [popped]))
                                    self.mark_path_failed(child_path)
                                    print(f"🔙 Going back one level (marked failed: {child_path})")
                                    
                                    # Check if we're back at root after going back
                                    if not self.current_path:
                                        self.back_to_root_count += 1
                                        print(f"🏠 Back at root level (count: {self.back_to_root_count})")
                                        
                                        # If we've gone back to root too many times, switch files
                                        if self.back_to_root_count >= 3:
                                            print("🔄 Excessive back-to-root navigation detected - switching to alternative file")
                                            return self._switch_to_alternative_file(user_input, step_count, max_steps)
                                else:
                                    print("❌ Already at root level")
                                    self.back_to_root_count += 1
                                    if self.back_to_root_count >= 3:
                                        print("🔄 Excessive back-to-root navigation detected - switching to alternative file")
                                        return self._switch_to_alternative_file(user_input, step_count, max_steps)
                                    return None
                            else:
                                print("🔄 Continuing exploration...")
                                
                        except Exception as e:
                            print(f"❌ Error in data analysis: {e}")
                            return None
                            
                    elif nav_decision['action'] == 'back':
                        # Go back one level with loop prevention
                        if self.current_path:
                            popped = self.current_path.pop()
                            child_path = ".".join(map(str, (self.current_path + [popped]) if self.current_path else [popped]))
                            self.mark_path_failed(child_path)
                            print(f"🔙 Going back one level (marked failed: {child_path})")
                            
                            # Check if we're back at root after going back
                            if not self.current_path:
                                self.back_to_root_count += 1
                                print(f"🏠 Back at root level (count: {self.back_to_root_count})")
                                
                                # If we've gone back to root too many times, switch files
                                if self.back_to_root_count >= 3:
                                    print("🔄 Excessive back-to-root navigation detected - switching to alternative file")
                                    return self._switch_to_alternative_file(user_input, step_count, max_steps)
                        else:
                            print("❌ Already at root level")
                            self.back_to_root_count += 1
                            if self.back_to_root_count >= 3:
                                print("🔄 Excessive back-to-root navigation detected - switching to alternative file")
                                return self._switch_to_alternative_file(user_input, step_count, max_steps)
                            return None
                            
                    elif nav_decision['action'] == 'complete':
                        # Found what we're looking for
                        current_path_str = ".".join(map(str, self.current_path)) if self.current_path else "root"
                        current_data = self._get_current_data()
                        print(f"🎉 Search complete! Found data at: {current_path_str}")
                        return (current_path_str, current_data)
                        
                except Exception as e:
                    print(f"❌ Error in navigation decision: {e}")
                    return None
                    
            elif context['current_level_type'] == 'list':
                print(f"📋 List with {context.get('list_length', 0)} items")
                # For lists, we can examine the structure or go back
                current_data = self._get_current_data()
                current_path_str = ".".join(map(str, self.current_path)) if self.current_path else "root"
                
                analysis_prompt = self.create_data_analysis_prompt(user_input, current_data, current_path_str)
                analysis_response = self.make_llm_call(analysis_prompt)
                
                try:
                    analysis_decision = self.parse_llm_response(analysis_response, "list_analysis")
                    print(f"📊 Analysis: {analysis_decision['reasoning']}")
                    
                    if analysis_decision['action'] == 'found':
                        print(f"🎉 Found relevant data at path: {current_path_str}")
                        return (current_path_str, analysis_decision.get('extracted_data', current_data))
                    else:
                        # Go back if list is not what we want
                        if self.current_path:
                            popped = self.current_path.pop()
                            child_path = ".".join(map(str, (self.current_path + [popped]) if self.current_path else [popped]))
                            self.mark_path_failed(child_path)
                            print(f"🔙 Going back one level (marked failed: {child_path})")
                            
                            # Check if we're back at root after going back
                            if not self.current_path:
                                self.back_to_root_count += 1
                                print(f"🏠 Back at root level (count: {self.back_to_root_count})")
                                
                                # If we've gone back to root too many times, switch files
                                if self.back_to_root_count >= 3:
                                    print("🔄 Excessive back-to-root navigation detected - switching to alternative file")
                                    return self._switch_to_alternative_file(user_input, step_count, max_steps)
                        else:
                            print("❌ Already at root level")
                            self.back_to_root_count += 1
                            if self.back_to_root_count >= 3:
                                print("🔄 Excessive back-to-root navigation detected - switching to alternative file")
                                return self._switch_to_alternative_file(user_input, step_count, max_steps)
                            return None
                            
                except Exception as e:
                    print(f"❌ Error in list analysis: {e}")
                    return None
                    
            else:
                # We're at a leaf value
                current_data = self._get_current_data()
                current_path_str = ".".join(map(str, self.current_path)) if self.current_path else "root"
                
                analysis_prompt = self.create_data_analysis_prompt(user_input, current_data, current_path_str)
                analysis_response = self.make_llm_call(analysis_prompt)
                
                try:
                    analysis_decision = self.parse_llm_response(analysis_response, "value_analysis")
                    print(f"📊 Analysis: {analysis_decision['reasoning']}")
                    
                    if analysis_decision['action'] == 'found':
                        print(f"🎉 Found relevant data at path: {current_path_str}")
                        return (current_path_str, current_data)
                    else:
                        # Go back
                        if self.current_path:
                            popped = self.current_path.pop()
                            child_path = ".".join(map(str, (self.current_path + [popped]) if self.current_path else [popped]))
                            self.mark_path_failed(child_path)
                            print(f"🔙 Going back one level (marked failed: {child_path})")
                            
                            # Check if we're back at root after going back
                            if not self.current_path:
                                self.back_to_root_count += 1
                                print(f"🏠 Back at root level (count: {self.back_to_root_count})")
                                
                                # If we've gone back to root too many times, switch files
                                if self.back_to_root_count >= 3:
                                    print("🔄 Excessive back-to-root navigation detected - switching to alternative file")
                                    return self._switch_to_alternative_file(user_input, step_count, max_steps)
                        else:
                            print("❌ Already at root level")
                            self.back_to_root_count += 1
                            if self.back_to_root_count >= 3:
                                print("🔄 Excessive back-to-root navigation detected - switching to alternative file")
                                return self._switch_to_alternative_file(user_input, step_count, max_steps)
                            return None
                            
                except Exception as e:
                    print(f"❌ Error in value analysis: {e}")
                    return None
        
        print(f"❌ Maximum search steps ({max_steps}) reached without finding target data")
        
        # If we've exhausted all paths in this file, ask LLM to choose another file
        print("🔄 All paths explored in current file. Asking LLM to choose another file...")
        
        # Reset path to root for new file exploration
        self.current_path = []
        
        # Ask LLM to select a different file
        file_selection_prompt = self.create_file_selection_prompt(user_input, self.json_files)
        file_response = self.make_llm_call(file_selection_prompt)
        
        try:
            file_decision = self.parse_llm_response(file_response, "alternative_file_selection")
            print(f"🤖 Alternative File Selection: {file_decision['reasoning']}")
            
            if file_decision['action'] == 'select_files':
                selected_indices = file_decision.get('choices', [])
                if selected_indices:
                    # Take the first alternative file
                    file_index = int(selected_indices[0]) - 1
                    if 0 <= file_index < len(self.json_files):
                        selected_file = self.json_files[file_index]
                        print(f"📂 Switching to alternative file: {selected_file}")
                        
                        # Load the alternative file
                        self.current_data = self.load_json_file(selected_file)
                        if not self.current_data:
                            print("❌ Failed to load alternative file")
                            return None
                        
                        # Track that we've explored this file too
                        self.explored_files.add(selected_file)
                        
                        # Recursively explore the new file with remaining steps
                        remaining_steps = max_steps - step_count
                        if remaining_steps > 0:
                            return self._explore_json_structure(user_input, selected_file)
                        else:
                            print("❌ No remaining steps for alternative file exploration")
                            return None
                    else:
                        print("❌ Invalid alternative file selection")
                        return None
                else:
                    print("❌ No alternative files selected")
                    return None
            else:
                print("❌ No alternative file selected")
                return None
                
        except Exception as e:
            print(f"❌ Error in alternative file selection: {e}")
            return None

    def _switch_to_alternative_file(self, user_input: str, current_step: int, max_steps: int) -> Optional[Tuple[str, Any]]:
        """Switch to an alternative file when loop prevention is triggered."""
        print("🔄 Switching to alternative file due to loop prevention...")
        
        # Reset path to root for new file exploration
        self.current_path = []
        
        # Ask LLM to select a different file
        file_selection_prompt = self.create_file_selection_prompt(user_input, self.json_files)
        file_response = self.make_llm_call(file_selection_prompt)
        
        try:
            file_decision = self.parse_llm_response(file_response, "alternative_file_selection")
            print(f"🤖 Alternative File Selection: {file_decision['reasoning']}")
            
            if file_decision['action'] == 'select_files':
                selected_indices = file_decision.get('choices', [])
                if selected_indices:
                    # Take the first alternative file
                    file_index = int(selected_indices[0]) - 1
                    if 0 <= file_index < len(self.json_files):
                        selected_file = self.json_files[file_index]
                        print(f"📂 Switching to alternative file: {selected_file}")
                        
                        # Load the alternative file
                        self.current_data = self.load_json_file(selected_file)
                        if not self.current_data:
                            print("❌ Failed to load alternative file")
                            return None
                        
                        # Track that we've explored this file too
                        self.explored_files.add(selected_file)
                        
                        # Reset loop prevention counters for new file
                        self.back_to_root_count = 0
                        self.path_visit_count = {}
                        
                        # Recursively explore the new file with remaining steps
                        remaining_steps = max_steps - current_step
                        if remaining_steps > 0:
                            return self._explore_json_structure(user_input, selected_file)
                        else:
                            print("❌ No remaining steps for alternative file exploration")
                            return None
                    else:
                        print("❌ Invalid alternative file selection")
                        return None
                else:
                    print("❌ No alternative files selected")
                    return None
            else:
                print("❌ No alternative file selected")
                return None
                
        except Exception as e:
            print(f"❌ Error in alternative file selection: {e}")
            return None

    def _get_current_data(self) -> Any:
        """Get data at current path in JSON structure."""
        current_data = self.current_data
        for path_part in self.current_path:
            if isinstance(current_data, dict) and path_part in current_data:
                current_data = current_data[path_part]
            elif isinstance(current_data, list) and isinstance(path_part, int) and path_part < len(current_data):
                current_data = current_data[path_part]
            else:
                return None
        return current_data
    
    def explore_json(self, user_input: str, start_path: str = None) -> Dict[str, Optional[Tuple[str, Any]]]:
        """Main method to explore JSON/YAML data based on user query."""
        print(f"🔍 AI JSON Explorer")
        print(f"=" * 50)
        print(f"Query: {user_input}")
        print(f"=" * 50)
        
        self.search_context = user_input
        self.conversation_history = []
        self.explored_files = set()  # Reset explored files for new session
        self.back_to_root_count = 0  # Reset loop prevention counters
        self.path_visit_count = {}   # Reset path visit tracking
        
        # Step 1: Get available JSON/YAML files
        if start_path is None:
            start_path = os.getcwd()
        
        self.json_files = self.get_available_json_files(start_path)
        if not self.json_files:
            print("❌ No JSON/YAML files found in the workspace")
            return {}
        
        print(f"📄 Found {len(self.json_files)} JSON/YAML files:")
        for f in self.json_files:
            print(f"  - {f}")

        
        # Step 2: Let LLM select which files to explore
        file_selection_prompt = self.create_file_selection_prompt(user_input, self.json_files)
        file_response = self.make_llm_call(file_selection_prompt)
        
        try:
            file_decision = self.parse_llm_response(file_response, "file_selection")
            print(f"🤖 Selected Files: {file_decision['reasoning']}")
            
            if file_decision['action'] == 'select_files':
                selected_indices = file_decision.get('choices', [])
                if not selected_indices:
                    print("❌ No files selected")
                    return {}
                
                results = {}
                for file_index in selected_indices:
                    file_index = int(file_index) - 1  # Convert to 0-based index
                    if 0 <= file_index < len(self.json_files):
                        selected_file = self.json_files[file_index]
                        print(f"\n📂 Exploring: {selected_file}")
                        
                        # Load the selected JSON/YAML file
                        self.current_data = self.load_json_file(selected_file)
                        if not self.current_data:
                            print("❌ Failed to load file")
                            results[selected_file] = None
                            continue
                        
                        # Track that we've explored this file
                        self.explored_files.add(selected_file)
                        
                        # Reset path for new file
                        self.current_path = []
                        self.back_to_root_count = 0
                        self.path_visit_count = {}
                        
                        # Explore the JSON structure
                        result = self._explore_json_structure(user_input, selected_file)
                        results[selected_file] = result
                        
                        if result:
                            print(f"✅ Found data in {selected_file}")
                        else:
                            print(f"❌ No relevant data found in {selected_file}")
                    else:
                        print(f"❌ Invalid file index: {file_index + 1}")
                
                return results
            else:
                print("❌ Unexpected file selection decision")
                return {}
                
        except Exception as e:
            print(f"❌ Error in file selection: {e}")
            return {}

def explore_json_data(user_input: str, start_path: str = None) -> Dict[str, Optional[Tuple[str, Any]]]:
    """Main entry point for JSON exploration."""
    print("AI-Powered JSON Explorer")
    print("=" * 50)

    explorer = AIJSONExplorer()
    results = explorer.explore_json(user_input, start_path)
    
    if results:
        print(f"\n🎉 Exploration completed for {len(results)} files:")
        for file_path, result in results.items():
            print(f"\n{'='*60}")
            print(f"📂 File: {file_path}")
            
            # Try to extract and display file description
            try:
                file_data = explorer.load_json_file(file_path)
                if isinstance(file_data, dict) and 'description' in file_data:
                    description = file_data['description']
                    print(f"📋 Description: {description}")
                else:
                    print("📋 Description: Not available")
            except Exception as e:
                print(f"� Description: Error loading - {str(e)}")
            
            if result:
                data_path, data_content = result
                print(f"📍 Data Path: {data_path}")
                print(f"📊 Data Type: {type(data_content).__name__}")
                
                # Show data preview
                data_preview = str(data_content)
                if len(data_preview) > 500:
                    data_preview = data_preview[:500] + "... [truncated]"
                print("📄 Content Preview:")
                # Pretty print JSON content
                if isinstance(data_content, (dict, list)):
                    print(json.dumps(data_content, indent=2))
                else:
                    print(data_content)
            else:
                print("❌ No relevant data found")
        
        return results
    else:
        print(f"\n❌ JSON exploration failed")
        return {}


if __name__ == "__main__":
    # Example usage
    user_inputs = [
        # "I need to find information on how to build a nuclear fusion plant in plexos such as classes and collections to use",
        "I have a Power2X object, but it should only be fed with green hydrogen. How can I ensure the hydrogen is green?"
    ]
    
    for user_input in user_inputs:
        print(f"\n{'='*80}")
        print(f"Testing query: {user_input}")
        print(f"{'='*80}")
        
        try:
            result = explore_json_data(user_input)
            if result:
                print("\n✅ Exploration completed successfully!")
            else:
                print("\n❌ Exploration completed without finding target data")
        except Exception as e:
            print(f"\n💥 Fatal error: {e}")
            traceback.print_exc()

        print("🏁 JSON exploration session complete")
        break  # Only test first query for now
