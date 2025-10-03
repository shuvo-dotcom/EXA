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
from typing import List, Dict, Any, Optional, Tuple
from src.tools.simple_file_browser import SimpleFileBrowser
import src.ai.file_operations.ai_finder_config as config
from src.ai.llm_calls.open_ai_calls import run_open_ai_ns as roains

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
                if file.lower().endswith(('.json', '.yaml', '.yml')):
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
        
        context = {
            "current_path": ".".join(map(str, self.current_path)) if self.current_path else "root",
            "current_level_type": type(current_level).__name__,
        }
        
        if isinstance(current_level, dict):
            context["available_keys"] = list(current_level.keys())[:20]  # Limit for token efficiency
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
        
        return context

    def parse_llm_response(self, response: str, context: str = "unknown") -> Dict[str, Any]:
        """Parse LLM response with better error handling and fallback."""
        if not response or not response.strip():
            print(f"⚠️ Empty response in {context}, using fallback")
            return {
                "action": "back",
                "reasoning": "Empty LLM response, going back to previous directory",
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
        response = roains(prompt, context)
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

    def get_directory_context(self, path: str) -> Dict[str, Any]:
        """Get context about current directory for LLM."""
        try:
            subdirectories = self.browser.get_subdirectories(path)
            all_files = self.browser.search_folder(path)
            supported_files = self.browser.filter_supported_files(all_files)
            
            context = {
                "current_path": path,
                "subdirectories": [Path(d).name for d in subdirectories],
                "file_count": len(all_files),
                "supported_files": [Path(f).name for f in supported_files[:10]],  # Limit for token efficiency
                "total_supported_files": len(supported_files)
            }
            
            return context
        except Exception as e:
            return {"error": str(e)}

    def create_file_selection_prompt(self, user_input: str, json_files: List[str]) -> str:
        """Create prompt for LLM to select which JSON/YAML file to explore."""
        system_prompt = config.SYSTEM_PROMPTS.get("navigation", "You are a helpful assistant.")
        
        prompt = f"""
                    USER QUERY: "{user_input}"

                    AVAILABLE JSON/YAML FILES:
                    {self.format_file_list(json_files)}

                    CONVERSATION HISTORY:
                    {self.format_conversation_history()}

                    TASK: Select the most relevant JSON/YAML file to explore based on the user's query.

                    Respond with a JSON object containing:
                    {{
                        "action": "select_file",
                        "reasoning": "Explain why this file is most relevant to the user's query", 
                        "choice": file_number,
                        "confidence": 0.0-1.0
                    }}

                    Choose the file number (1-{len(json_files)}) that seems most relevant to the user's query.
                    """
        return prompt

    def create_navigation_prompt(self, user_input: str, context: Dict[str, Any]) -> str:
        """Create prompt for LLM to decide JSON navigation."""
        system_prompt = config.SYSTEM_PROMPTS.get("navigation", "You are a helpful assistant.")
        
        prompt = f"""
                    USER QUERY: "{user_input}"

                    CURRENT JSON CONTEXT:
                    - Current path: {context.get('current_path', 'root')}
                    - Current level type: {context.get('current_level_type', 'Unknown')}
                    - Available keys: {context.get('available_keys', [])}
                    - Total keys: {context.get('total_keys', 0)}
                    - Structure preview: {context.get('structure_preview', {})}

                    DECISION HISTORY AND LEARNING:
                    {self.format_conversation_history()}

                    FAILURE PREVENTION:
                    {self.format_failure_prevention_context()}

                    TASK: Analyze the JSON structure and decide the next step to find what the user is looking for.

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

    def create_file_evaluation_prompt(self, user_input: str, file_path: str, file_content: Dict[str, Any]) -> str:
        """Create prompt for LLM to evaluate if file is correct."""
        system_prompt = config.SYSTEM_PROMPTS["content_evaluation"]
        content_summary = self.summarize_file_content(file_content)
        
        prompt = f"""
                    USER QUERY: "{user_input}"

                    FILE BEING EVALUATED:
                    - Path: {file_path}
                    - Name: {Path(file_path).name}
                    - Type: {file_content.get('type', 'unknown')}

                    CONTENT SUMMARY:
                    {content_summary}

                    CONVERSATION HISTORY:
                    {self.format_conversation_history()}

                    TASK: Determine if this file matches what the user is looking for.

                    Respond with a JSON object:
                    {{
                        "action": "complete|continue",
                        "reasoning": "Why this file does/doesn't match the requirements",
                        "is_correct": true|false,
                        "confidence": 0.0-1.0,
                        "next_step": "description of what to do next if not correct"
                    }}

                    If is_correct is true, action should be "complete"
                    If is_correct is false, action should be "continue"
                    """
        return prompt

    def format_conversation_history(self) -> str:
        """Format comprehensive conversation history for LLM context."""
        if not self.decision_history:
            return "No previous decisions made."
        
        history_lines = []
        
        # Add summary of visited paths
        if self.visited_paths:
            history_lines.append(f"VISITED PATHS: {len(self.visited_paths)} directories explored")
            recent_paths = list(self.visited_paths)[-3:]  # Last 3 paths
            for path in recent_paths:
                history_lines.append(f"  - {Path(path).name}")
        
        # Add rejected files
        if self.rejected_files:
            history_lines.append(f"\nREJECTED FILES: {len(self.rejected_files)} files examined but not suitable")
            for rejected in self.rejected_files[-3:]:  # Last 3 rejected
                history_lines.append(f"  - {rejected['file']} (Reason: {rejected['reason']})")
        
        # Add recent decisions with reasoning
        history_lines.append(f"\nRECENT DECISIONS:")
        recent_decisions = self.decision_history[-5:]  # Last 5 decisions
        
        for decision in recent_decisions:
            action = decision['decision'].get('action', 'unknown')
            choice = decision['decision'].get('choice', 'none')
            reasoning = decision['reasoning'][:100] + "..." if len(decision['reasoning']) > 100 else decision['reasoning']
            
            history_lines.append(f"Step {decision['step']}: {action} -> {choice}")
            history_lines.append(f"  Path: {Path(decision['path']).name}")
            history_lines.append(f"  Reasoning: {reasoning}")
            history_lines.append(f"  Confidence: {decision['confidence']}")
            
            # Add outcome if available
            if decision['type'] == 'navigation':
                if choice in [d.name for d in Path(decision['path']).iterdir() if d.is_dir()]:
                    history_lines.append(f"  Outcome: ✅ Successfully navigated")
                else:
                    history_lines.append(f"  Outcome: ❌ Directory not found")
            history_lines.append("")
        
        return "\n".join(history_lines)

    def format_failure_prevention_context(self) -> str:
        """Format context about paths and decisions to avoid."""
        prevention_lines = []
        
        # Track paths that led to failures in this attempt
        failed_navigation_paths = set()
        for decision in self.decision_history:
            if decision.get('type') in ['navigation_failed']:
                path = decision.get('path')
                choice = decision.get('decision', {}).get('choice')
                if path and choice:
                    failed_navigation_paths.add(f"{Path(path).name} -> {choice}")
        
        if failed_navigation_paths:
            prevention_lines.append("PATHS THAT FAILED IN THIS ATTEMPT:")
            for failed_path in failed_navigation_paths:
                prevention_lines.append(f"  ❌ {failed_path}")
        
        # Track directories we keep visiting without success
        path_visit_count = {}
        for path in self.visited_paths:
            path_name = Path(path).name
            path_visit_count[path_name] = path_visit_count.get(path_name, 0) + 1
        
        repeated_paths = {k: v for k, v in path_visit_count.items() if v > 1}
        if repeated_paths:
            prevention_lines.append("\nREPEATEDLY VISITED PATHS (potentially problematic):")
            for path, count in repeated_paths.items():
                prevention_lines.append(f"  🔄 {path} (visited {count} times)")
        
        # Global failure knowledge (if available)
        if hasattr(self, 'global_failed_paths') and self.global_failed_paths:
            prevention_lines.append("\nGLOBAL FAILED PATHS (from previous attempts):")
            for failed_path in list(self.global_failed_paths)[-3:]:  # Show last 3
                prevention_lines.append(f"  🚫 {Path(failed_path).name}")
                
        if hasattr(self, 'attempt_number') and self.attempt_number > 1:
            prevention_lines.append(f"\nATTEMPT {self.attempt_number}: Previous attempts failed, try different approaches!")
        
        return "\n".join(prevention_lines) if prevention_lines else "No specific failures to avoid yet."

    def log_decision(self, decision_type: str, decision: Dict[str, Any], context: Dict[str, Any]):
        """Log LLM decision with full context for better history tracking."""
        decision_entry = {
            "step": len(self.decision_history) + 1,
            "type": decision_type,  # "navigation", "file_selection", "file_evaluation"
            "path": self.current_path,
            "decision": decision,
            "context": {
                "subdirectories": context.get('subdirectories', []),
                "files": context.get('supported_files', []),
                "total_files": context.get('total_supported_files', 0)
            },
            "reasoning": decision.get('reasoning', ''),
            "confidence": decision.get('confidence', 0.0)
        }
        self.decision_history.append(decision_entry)
        
        # Add to visited paths if navigating
        if decision.get('action') == 'navigate' and decision.get('choice'):
            target_path = Path(self.current_path) / decision.get('choice')
            self.visited_paths.add(str(target_path))

    def is_repeating_failed_action(self, current_decision: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if the current decision repeats a previously failed action."""
        if not self.decision_history:
            return False
        
        current_action = current_decision.get('action')
        current_choice = current_decision.get('choice')
        current_path = self.current_path
        
        # Check recent decisions for similar actions at the same location
        for decision in self.decision_history[-3:]:  # Check last 3 decisions
            if (decision['path'] == current_path and 
                decision['decision'].get('action') == current_action and
                decision['decision'].get('choice') == current_choice):
                
                # Check if it failed (no successful navigation after it)
                decision_index = self.decision_history.index(decision)
                subsequent_decisions = self.decision_history[decision_index + 1:]
                
                # If we're back at the same location with same decision, it likely failed
                if any(d['path'] == current_path for d in subsequent_decisions):
                    return True
        
        return False

    def suggest_alternative_action(self, failed_decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest an alternative action when the current one has failed before."""
        action = failed_decision.get('action')
        choice = failed_decision.get('choice')
        
        # If navigation failed, try examining files or going back
        if action == 'navigate':
            if context.get('total_supported_files', 0) > 0:
                return {
                    "action": "examine",
                    "choice": "files",
                    "reasoning": f"Navigation to '{choice}' failed before. Examining files in current directory instead.",
                    "confidence": 0.6
                }
            else:
                return {
                    "action": "back",
                    "choice": "back",
                    "reasoning": f"Navigation to '{choice}' failed before and no files to examine. Going back to parent directory.",
                    "confidence": 0.7
                }
        
        # If examination failed, try navigating or going back
        elif action == 'examine':
            subdirs = context.get('subdirectories', [])
            if subdirs:
                # Choose a different subdirectory than previously tried
                tried_dirs = {d['decision'].get('choice') for d in self.decision_history 
                             if d['decision'].get('action') == 'navigate' and d['path'] == self.current_path}
                available_dirs = [d for d in subdirs if d not in tried_dirs]
                
                if available_dirs:
                    return {
                        "action": "navigate",
                        "choice": available_dirs[0],
                        "reasoning": f"File examination failed before. Trying unexplored directory '{available_dirs[0]}'.",
                        "confidence": 0.6
                    }
            
            return {
                "action": "back",
                "choice": "back", 
                "reasoning": "File examination failed before and no new directories to explore. Going back.",
                "confidence": 0.7
            }
        
        # Default: return original decision
        return failed_decision

    def should_avoid_path(self, target_path: str) -> bool:
        """Check if we should avoid a particular path based on history."""
        path_name = Path(target_path).name.lower()
        
        # Check if this exact path has failed before
        if hasattr(self, 'global_failed_paths'):
            for failed_path in self.global_failed_paths:
                if Path(failed_path).name.lower() == path_name:
                    return True
        
        # Check if we've visited this path multiple times without success
        visit_count = sum(1 for visited in self.visited_paths 
                         if Path(visited).name.lower() == path_name)
        if visit_count >= 2:
            return True
            
        return False
    
    def get_alternative_directories(self, subdirectories: List[str], context: Dict[str, Any]) -> List[str]:
        """Get alternative directories prioritizing unexplored ones."""
        if not subdirectories:
            return []
        
        # Separate directories into explored vs unexplored
        explored = []
        unexplored = []
        
        for subdir in subdirectories:
            path_name = Path(subdir).name.lower()
            
            # Check if we've been to this directory before
            visited_this_dir = any(Path(visited).name.lower() == path_name 
                                 for visited in self.visited_paths)
            
            # Check if this directory failed before
            should_avoid = self.should_avoid_path(subdir)
            
            if should_avoid:
                continue  # Skip completely failed paths
            elif visited_this_dir:
                explored.append(subdir)
            else:
                unexplored.append(subdir)
        
        # Prioritize unexplored, then explored
        return unexplored + explored

    def is_repeating_failed_action(self, decision: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if the current decision repeats a previously failed action."""
        if not self.decision_history:
            return False
        
        current_action = decision.get('action')
        current_choice = decision.get('choice')
        current_path = self.current_path
        
        # Check recent decisions for similar actions at the same location
        for decision_entry in self.decision_history[-3:]:  # Check last 3 decisions
            if (decision_entry['path'] == current_path and 
                decision_entry['decision'].get('action') == current_action and
                decision_entry['decision'].get('choice') == current_choice):
                
                # Check if it failed (no successful navigation after it)
                decision_index = self.decision_history.index(decision_entry)
                subsequent_decisions = self.decision_history[decision_index + 1:]
                
                # If we're back at the same location with same decision, it likely failed
                if any(d['path'] == current_path for d in subsequent_decisions):
                    return True
        
        return False

    def format_file_list(self, files: List[str]) -> str:
        """Format file list for LLM."""
        formatted = []
        for i, file_path in enumerate(files, 1):
            file_obj = Path(file_path)
            size = self.browser.get_file_size(file_path)
            formatted.append(f"{i}. {file_obj.name} ({file_obj.suffix}, {size})")
        
        return "\n".join(formatted)

    def summarize_file_content(self, file_content: Dict[str, Any]) -> str:
        """Summarize file content for LLM evaluation."""
        if file_content.get('type') == 'error':
            return f"Error reading file: {file_content.get('message', 'Unknown error')}"
        
        content_type = file_content.get('type', 'unknown')
        summary = [f"File type: {content_type}"]
        
        if content_type == 'text':
            lines = file_content.get('lines', 0)
            chars = file_content.get('characters', 0)
            content = file_content.get('content', '')
            preview = content[:500] + "..." if len(content) > 500 else content
            summary.extend([
                f"Lines: {lines}",
                f"Characters: {chars}",
                f"Preview: {preview}"
            ])
        
        elif content_type == 'csv':
            summary.extend([
                f"Rows: {file_content.get('total_rows', 0)}",
                f"Columns: {file_content.get('total_columns', 0)}",
                f"Headers: {', '.join(file_content.get('headers', [])[:5])}"
            ])
        
        elif content_type == 'json':
            keys = file_content.get('keys', [])
            if keys:
                summary.append(f"Top-level keys: {', '.join(keys[:5])}")
            summary.append(f"Size: {file_content.get('size', 0)} characters")
        
        return "\n".join(summary)

    def log_step(self, step_description: str):
        """Log a step in the conversation history."""
        self.conversation_history.append(step_description)
        print(f"📝 {step_description}")

    def explore_json(self, user_input: str, start_path: str = None) -> Optional[Tuple[str, Any]]:
        """Main method to explore JSON/YAML data based on user query."""
        print(f"🔍 AI JSON Explorer")
        print(f"=" * 50)
        print(f"Query: {user_input}")
        print(f"=" * 50)
        
        self.search_context = user_input
        self.conversation_history = []
        
        # Step 1: Get available JSON/YAML files
        if start_path is None:
            start_path = os.getcwd()
        
        self.json_files = self.get_available_json_files(start_path)
        if not self.json_files:
            print("❌ No JSON/YAML files found in the workspace")
            return None
        
        print(f"📄 Found {len(self.json_files)} JSON/YAML files")
        
        # Step 2: Let LLM select which file to explore
        file_selection_prompt = self.create_file_selection_prompt(user_input, self.json_files)
        file_response = self.make_llm_call(file_selection_prompt)
        
        try:
            file_decision = self.parse_llm_response(file_response, "file_selection")
            print(f"🤖 Selected File: {file_decision['reasoning']}")
            
            if file_decision['action'] == 'select_file':
                file_index = int(file_decision['choice']) - 1
                if 0 <= file_index < len(self.json_files):
                    selected_file = self.json_files[file_index]
                    print(f"📂 Loading: {selected_file}")
                    
                    # Load the selected JSON/YAML file
                    self.current_data = self.load_json_file(selected_file)
                    if not self.current_data:
                        print("❌ Failed to load file")
                        return None
                    
                    # Step 3: Explore the JSON structure
                    self.current_path = []
                    return self._explore_json_structure(user_input, selected_file)
                else:
                    print("❌ Invalid file selection")
                    return None
            else:
                print("❌ Unexpected file selection decision")
                return None
                
        except Exception as e:
            print(f"❌ Error in file selection: {e}")
            return None

    def _explore_json_structure(self, user_input: str, file_path: str) -> Optional[Tuple[str, Any]]:
        """Explore JSON structure to find relevant data."""
        max_steps = config.MAX_SEARCH_STEPS
        step_count = 0
        
        while step_count < max_steps:
            step_count += 1
            print(f"\n🔄 Step {step_count}")
            
            # Get current level context
            context = self.get_current_level_context()
            
            if context.get('error'):
                print(f"❌ Error accessing data: {context['error']}")
                return None
            
            print(f"� Current path: {context['current_path']}")
            print(f"� Level type: {context['current_level_type']}")
            
            if context['current_level_type'] == 'dict':
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
                                    self.current_path.pop()
                                    print("🔙 Going back one level")
                                else:
                                    print("❌ Already at root level")
                                    return None
                            else:
                                print("🔄 Continuing exploration...")
                                
                        except Exception as e:
                            print(f"❌ Error in data analysis: {e}")
                            return None
                            
                    elif nav_decision['action'] == 'back':
                        # Go back one level
                        if self.current_path:
                            self.current_path.pop()
                            print("🔙 Going back one level")
                        else:
                            print("❌ Already at root level")
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
                            self.current_path.pop()
                            print("🔙 Going back one level")
                        else:
                            print("❌ Already at root level")
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
                            self.current_path.pop()
                            print("🔙 Going back one level")
                        else:
                            print("❌ Already at root level")
                            return None
                            
                except Exception as e:
                    print(f"❌ Error in value analysis: {e}")
                    return None
        
        print(f"❌ Maximum search steps ({max_steps}) reached without finding target data")
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

    def format_conversation_history(self) -> str:
                            # Ask LLM to choose which file to examine
                            file_prompt = self.create_file_analysis_prompt(user_input, supported_files)
                            file_response = self.make_llm_call(file_prompt)
                            
                            try:
                                file_decision = self.parse_llm_response(file_response, "file_selection")
                                
                                # Log file selection decision
                                file_context = {"supported_files": [Path(f).name for f in supported_files]}
                                self.log_decision("file_selection", file_decision, file_context)
                                
                                file_choice = file_decision.get('choice', 1)
                                
                                if isinstance(file_choice, int) and 1 <= file_choice <= len(supported_files):
                                    selected_file = supported_files[file_choice - 1]
                                    
                                    self.log_step(f"Examining file: {Path(selected_file).name}")
                                    
                                    # Read and analyze the file
                                    file_ext = Path(selected_file).suffix.lower()
                                    if file_ext in self.browser.supported_formats:
                                        file_content = self.browser.supported_formats[file_ext](selected_file)
                                        
                                        # Display file content
                                        print(f"\n📖 File Content Preview:")
                                        self.browser.display_file_content(file_content, selected_file)
                                        
                                        # Ask LLM if this is the correct file
                                        eval_prompt = self.create_file_evaluation_prompt(user_input, selected_file, file_content)
                                        eval_response = self.make_llm_call(eval_prompt)
                                        
                                        try:
                                            eval_decision = self.parse_llm_response(eval_response, "file_evaluation")
                                            
                                            # Log file evaluation decision
                                            eval_context = {"file_path": selected_file, "file_type": file_content.get('type')}
                                            self.log_decision("file_evaluation", eval_decision, eval_context)
                                            
                                            print(f"\n🤖 Evaluation: {eval_decision['reasoning']}")
                                            
                                            if eval_decision.get('is_correct', False):
                                                print(f"✅ Found the correct file!")
                                                return selected_file, file_content
                                            else:
                                                # Add to rejected files list
                                                self.rejected_files.append({
                                                    "file": Path(selected_file).name,
                                                    "path": selected_file,
                                                    "reason": eval_decision['reasoning'][:100],
                                                    "step": len(self.decision_history)
                                                })
                                                
                                                print(f"❌ Not the right file. {eval_decision.get('next_step', 'Continuing search...')}")
                                                self.log_step(f"File {Path(selected_file).name} not correct - {eval_decision['reasoning']}")
                                                
                                                # Ask user if they want to continue or manually override
                                                # user_input = input("\nContinue search? (y/n/manual): ").strip().lower()
                                                # if user_input == 'n':
                                                #     return None
                                                # elif user_input == 'manual':
                                                #     return self.manual_file_selection(supported_files)

                                        except Exception as e:
                                            print(f"❌ Error parsing LLM evaluation response: {e}")
                                            return None
                                            
                            except Exception as e:
                                traceback.print_exc()
                                print(f"❌ Error parsing LLM file selection response: {e}")
                                return None
                    
                    elif nav_decision['action'] == 'navigate':
                        # Navigate to chosen directory
                        choice = nav_decision.get('choice', '')
                        subdirectories = self.browser.get_subdirectories(self.current_path)
                        
                        # Check if the chosen directory should be avoided
                        target_path = None
                        for subdir in subdirectories:
                            if Path(subdir).name.lower() == choice.lower():
                                target_path = subdir
                                break
                        
                        # If the choice should be avoided, suggest alternatives
                        if target_path and self.should_avoid_path(target_path):
                            print(f"⚠️ Avoiding previously failed path: {choice}")
                            alternative_dirs = self.get_alternative_directories(subdirectories, context)
                            if alternative_dirs:
                                chosen_dir = alternative_dirs[0]
                                print(f"🔄 Choosing alternative: {Path(chosen_dir).name}")
                            else:
                                print("❌ No alternative directories available")
                                return None
                        else:
                            # Find matching directory
                            chosen_dir = None
                            for subdir in subdirectories:
                                if Path(subdir).name.lower() == choice.lower():
                                    chosen_dir = subdir
                                    break
                        
                        if chosen_dir:
                            self.current_path = chosen_dir
                            self.log_step(f"Navigated to: {Path(chosen_dir).name}")
                        else:
                            print(f"❌ Directory '{choice}' not found")
                            # List available directories for debugging
                            available_dirs = [Path(d).name for d in subdirectories]
                            print(f"Available directories: {available_dirs}")
                            
                            # Log failed navigation attempt
                            failed_context = {
                                "attempted_choice": choice,
                                "available_directories": available_dirs,
                                "current_path": self.current_path
                            }
                            failed_decision = {
                                "action": "navigate", 
                                "choice": choice,
                                "reasoning": f"Directory '{choice}' not found",
                                "confidence": 0.0
                            }
                            self.log_decision("navigation_failed", failed_decision, failed_context)
                            return None
                    
                    elif nav_decision['action'] == 'back':
                        # Go back to parent directory
                        parent_path = Path(self.current_path).parent
                        if str(parent_path) != self.current_path:
                            old_path = self.current_path
                            self.current_path = str(parent_path)
                            self.log_step(f"Went back to: {Path(parent_path).name}")
                            
                            # Log back navigation
                            back_context = {
                                "from_path": old_path,
                                "to_path": self.current_path,
                                "reason": nav_decision.get('reasoning', 'Back navigation requested')
                            }
                            back_decision = {
                                "action": "back",
                                "choice": "back", 
                                "reasoning": nav_decision.get('reasoning', 'Back navigation requested'),
                                "confidence": nav_decision.get('confidence', 1.0)
                            }
                            self.log_decision("back_navigation", back_decision, back_context)
                        else:
                            print("❌ Already at root directory")
                            # Log failed back attempt
                            failed_back_context = {
                                "current_path": self.current_path,
                                "reason": "Already at root directory"
                            }
                            failed_back_decision = {
                                "action": "back",
                                "choice": "back",
                                "reasoning": "Already at root directory",
                                "confidence": 0.0
                            }
                            self.log_decision("back_navigation_failed", failed_back_decision, failed_back_context)
                            return None
                    
                    elif nav_decision['action'] == 'complete':
                        print("✅ Search completed by LLM decision")
                        nav_decision['filepath'] = context['current_path']
                        return nav_decision

                except Exception as e:
                    traceback.print_exc()
                    print(f"❌ Error parsing LLM navigation response: {e}")
                    return None
            
            else:
                # No files in current directory, navigate to subdirectories
                subdirectories = self.browser.get_subdirectories(self.current_path)
                if subdirectories:
                    nav_prompt = self.create_navigation_prompt(user_input, context)
                    nav_response = self.make_llm_call(nav_prompt)
                    
                    try:
                        nav_decision = self.parse_llm_response(nav_response, "navigation_no_files")
                        
                        # Log the navigation decision
                        nav_context = {
                            "has_files": False,
                            "subdirectories": [Path(d).name for d in subdirectories],
                            "current_path": self.current_path
                        }
                        self.log_decision("navigation_no_files", nav_decision, nav_context)
                        
                        choice = nav_decision.get('choice', '')
                        
                        # Handle back action first
                        if nav_decision.get('action') == 'back':
                            parent_path = Path(self.current_path).parent
                            if str(parent_path) != self.current_path:
                                old_path = self.current_path
                                self.current_path = str(parent_path)
                                self.log_step(f"Went back to: {Path(parent_path).name}")
                                
                                # Log back navigation
                                back_context = {
                                    "from_path": old_path,
                                    "to_path": self.current_path,
                                    "reason": nav_decision.get('reasoning', 'Back navigation requested'),
                                    "context": "no_files_in_directory"
                                }
                                back_decision = {
                                    "action": "back",
                                    "choice": "back",
                                    "reasoning": nav_decision.get('reasoning', 'Back navigation requested'),
                                    "confidence": nav_decision.get('confidence', 1.0)
                                }
                                self.log_decision("back_navigation", back_decision, back_context)
                            else:
                                print("❌ Already at root directory")
                                # Log failed back attempt
                                failed_back_context = {
                                    "current_path": self.current_path,
                                    "reason": "Already at root directory",
                                    "context": "no_files_in_directory"
                                }
                                failed_back_decision = {
                                    "action": "back",
                                    "choice": "back",
                                    "reasoning": "Already at root directory",
                                    "confidence": 0.0
                                }
                                self.log_decision("back_navigation_failed", failed_back_decision, failed_back_context)
                                return None
                        else:
                            # Navigation to chosen directory  
                            choice = nav_decision.get('choice', '')
                            subdirectories = self.browser.get_subdirectories(self.current_path)
                            
                            # Check if the chosen directory should be avoided
                            target_path = None
                            for subdir in subdirectories:
                                if Path(subdir).name.lower() == choice.lower():
                                    target_path = subdir
                                    break
                            
                            # If the choice should be avoided, suggest alternatives
                            if target_path and self.should_avoid_path(target_path):
                                print(f"⚠️ Avoiding previously failed path: {choice}")
                                alternative_dirs = self.get_alternative_directories(subdirectories, context)
                                if alternative_dirs:
                                    chosen_dir = alternative_dirs[0]
                                    print(f"🔄 Choosing alternative: {Path(chosen_dir).name}")
                                else:
                                    print("❌ No alternative directories available")
                                    return None
                            else:
                                # Find matching directory
                                chosen_dir = None
                                for subdir in subdirectories:
                                    if Path(subdir).name.lower() == choice.lower():
                                        chosen_dir = subdir
                                        break
                            
                            if chosen_dir:
                                self.current_path = chosen_dir
                                self.log_step(f"Navigated to: {Path(chosen_dir).name}")
                                
                                # Log successful navigation
                                success_context = {
                                    "chosen_directory": Path(chosen_dir).name,
                                    "available_directories": [Path(d).name for d in subdirectories],
                                    "current_path": self.current_path,
                                    "context": "no_files_in_directory"
                                }
                                success_decision = {
                                    "action": "navigate",
                                    "choice": choice,
                                    "reasoning": nav_decision.get('reasoning', 'Navigated to directory'),
                                    "confidence": nav_decision.get('confidence', 1.0)
                                }
                                self.log_decision("navigation_success", success_decision, success_context)
                            else:
                                print(f"❌ Directory '{choice}' not found")
                                # List available directories for debugging
                                available_dirs = [Path(d).name for d in subdirectories]
                                print(f"Available directories: {available_dirs}")
                                
                                # Log failed navigation attempt
                                failed_context = {
                                    "attempted_choice": choice,
                                    "available_directories": available_dirs,
                                    "current_path": self.current_path,
                                    "context": "no_files_in_directory"
                                }
                                failed_decision = {
                                    "action": "navigate",
                                    "choice": choice,
                                    "reasoning": f"Directory '{choice}' not found",
                                    "confidence": 0.0
                                }
                                self.log_decision("navigation_failed", failed_decision, failed_context)
                                return None

                    except Exception as e:
                        print(f"❌ Error parsing LLM response: {e}")
                        return None
                else:
                    print("❌ No subdirectories or files found")
                    # Automatically go back to parent directory instead of ending
                    parent_path = Path(self.current_path).parent
                    if str(parent_path) != self.current_path:
                        old_path = self.current_path
                        self.current_path = str(parent_path)
                        self.log_step(f"Auto-navigating back to: {Path(parent_path).name}")
                        
                        # Log automatic back navigation
                        back_context = {
                            "from_path": old_path,
                            "to_path": self.current_path,
                            "reason": "Empty directory - automatically going back",
                            "context": "auto_back_navigation"
                        }
                        back_decision = {
                            "action": "back",
                            "choice": "back",
                            "reasoning": "Empty directory - automatically going back to continue search",
                            "confidence": 1.0
                        }
                        self.log_decision("auto_back_navigation", back_decision, back_context)
                        continue  # Continue the search loop
                    else:
                        print("❌ Already at root directory and no files found")
                        return None
        
        print(f"❌ Maximum steps ({max_steps}) reached")
        return None

    def manual_file_selection(self, files: List[str]) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Allow manual file selection as fallback."""
        print("\n🔧 Manual file selection mode")
        self.browser.display_file_list(files)
        
        while True:
            try:
                choice = input(f"Select a file (1-{len(files)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return None
                
                index = int(choice) - 1
                if 0 <= index < len(files):
                    selected_file = files[index]
                    file_ext = Path(selected_file).suffix.lower()
                    if file_ext in self.browser.supported_formats:
                        file_content = self.browser.supported_formats[file_ext](selected_file)
                        self.browser.display_file_content(file_content, selected_file)
                        
                        confirm = input("Is this the correct file? (y/n): ").strip().lower()
                        if confirm == 'y':
                            return selected_file, file_content
                    
            except Exception as e:
                traceback.print_exc()
                print(f"Please enter a valid number: {e}")

def find_starting_path(user_input: str) -> Optional[str]:
    """Find the starting path based on the user query."""
    file_locations = r'config\file_locations.json'
    with open(file_locations, 'r') as f:
        locations = json.load(f)


    file_starting_location_prompt = f"""
    You are an agent who's task it is to find the correct root path for a given user query, 
    from a list of options. This will be passed to another agent to continue traversing the file structure.
    Based on the user query, you should return the most relevant model location.
    Here is the user query: {user_input}
    Here are the available model locations: {locations}. 

    Please return the most relevant location verbatim as a json do not modify the filename as that directory may not exist. in the following format:
    {{
        "root_path": "<path_to_file>",
        "reasoning": "<reasoning>"
    }}
    """

    response = roains(file_starting_location_prompt, context)
    try:
        response_json = json.loads(response)
        return response_json.get("root_path")
    except json.JSONDecodeError as e:
        print(f"⚠️ Error parsing starting path response: {e}")
        # Fallback to a reasonable default path
        return r"C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Sectoral Model"

def main(user_input, open_file = False):
    """Main entry point."""
    print("AI-Powered File Finder")
    print("=" * 30)

    start_path = find_starting_path(user_input)

    # Initialize and run AI file finder
    finder = AIFileFinder()
    result = finder.find_file(user_input, start_path)
    
    if result:
        file_location = result[0]
        file_content = result[1]
        print(f"\n🎉 Successfully found file:")
        print(f"📁 Path: {file_location}")
        print(f"📄 Name: {Path(file_location).name}")

        # Option to open externally
        if open_file:
            finder.browser.open_file_externally(file_location)
            os.startfile(file_location)
        return file_location, file_content
    else:
        print("\n❌ File search was not successful")


class AIFileFinderWithRetry:
    """Enhanced AI File Finder with retry mechanism that learns from previous attempts."""
    
    def __init__(self):
        """Initialize the retry system."""
        self.finder = None
        self.global_failed_paths = set()  # Paths that failed across all attempts
        self.global_rejected_files = []   # Files rejected across all attempts
        self.global_failed_decisions = [] # Failed decisions across all attempts
        self.attempt_histories = []       # Store history from each attempt
        
    def should_avoid_decision(self, decision: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if this decision should be avoided based on previous attempt failures."""
        action = decision.get('action')
        choice = decision.get('choice')
        current_path = context.get('current_path', '')
        
        # Check if this exact decision failed before
        for failed_decision in self.global_failed_decisions:
            if (failed_decision.get('action') == action and 
                failed_decision.get('choice') == choice and 
                failed_decision.get('path') == current_path):
                return True
        
        # Check if this path was problematic before
        if action == 'navigate':
            target_path = Path(current_path) / choice if choice else None
            if target_path and str(target_path) in self.global_failed_paths:
                return True
        
        # Check if this file was rejected before
        if action == 'examine' and choice in [f['file'] for f in self.global_rejected_files]:
            return True
            
        return False
    
    def suggest_alternative_from_history(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Suggest alternative decision based on global failure history."""
        action = decision.get('action')
        current_path = context.get('current_path', '')
        
        if action == 'navigate':
            # Find directories that haven't failed before
            available_dirs = context.get('subdirectories', [])
            failed_dirs = {Path(fp).name for fp in self.global_failed_paths}
            safe_dirs = [d for d in available_dirs if d not in failed_dirs]
            
            if safe_dirs:
                return {
                    "action": "navigate",
                    "choice": safe_dirs[0],
                    "reasoning": f"Avoiding previously failed paths. Trying {safe_dirs[0]} instead.",
                    "confidence": 0.7
                }
        
        elif action == 'examine':
            # If examination failed before, try navigation instead
            available_dirs = context.get('subdirectories', [])
            if available_dirs:
                failed_dirs = {Path(fp).name for fp in self.global_failed_paths}
                safe_dirs = [d for d in available_dirs if d not in failed_dirs]
                
                if safe_dirs:
                    return {
                        "action": "navigate", 
                        "choice": safe_dirs[0],
                        "reasoning": "File examination failed in previous attempts. Trying navigation instead.",
                        "confidence": 0.6
                    }
        
        # If navigation failed, try going back
        return {
            "action": "back",
            "choice": "back",
            "reasoning": "Previous attempts suggest this path is not productive. Going back.",
            "confidence": 0.8
        }
    
    def update_global_failures(self, finder_instance: AIFileFinder):
        """Update global failure tracking based on completed attempt."""
        if not finder_instance:
            return
            
        # Add failed navigation paths
        for decision in finder_instance.decision_history:
            if decision.get('type') in ['navigation_failed', 'back_navigation_failed']:
                path = decision.get('path', '')
                choice = decision.get('decision', {}).get('choice', '')
                
                if path and choice and choice != 'back':
                    failed_path = Path(path) / choice
                    self.global_failed_paths.add(str(failed_path))
                    
                # Store the failed decision
                failed_decision = {
                    'action': decision.get('decision', {}).get('action'),
                    'choice': decision.get('decision', {}).get('choice'),
                    'path': path,
                    'reasoning': decision.get('reasoning', ''),
                    'attempt': len(self.attempt_histories)
                }
                self.global_failed_decisions.append(failed_decision)
        
        # Add rejected files
        for rejected_file in finder_instance.rejected_files:
            if rejected_file not in self.global_rejected_files:
                rejected_file['attempt'] = len(self.attempt_histories)
                self.global_rejected_files.append(rejected_file)
        
        # Store this attempt's history
        self.attempt_histories.append({
            'attempt': len(self.attempt_histories) + 1,
            'decision_history': finder_instance.decision_history.copy(),
            'visited_paths': finder_instance.visited_paths.copy(),
            'rejected_files': finder_instance.rejected_files.copy()
        })
    
    def create_enhanced_finder(self, attempt_number: int) -> AIFileFinder:
        """Create a new AIFileFinder instance enhanced with global failure knowledge."""
        finder = AIFileFinder()
        
        # Pre-populate with global failure knowledge
        finder.global_failed_paths = self.global_failed_paths.copy()
        finder.global_rejected_files = self.global_rejected_files.copy()
        finder.global_failed_decisions = self.global_failed_decisions.copy()
        finder.attempt_number = attempt_number
        
        # Enhance the finder's decision-making methods
        original_create_navigation_prompt = finder.create_navigation_prompt
        
        def enhanced_navigation_prompt(user_input: str, context: Dict[str, Any]) -> str:
            """Enhanced navigation prompt that includes global failure history."""
            base_prompt = original_create_navigation_prompt(user_input, context)
            
            if attempt_number > 1:
                failure_context = f"""
GLOBAL FAILURE HISTORY (from {attempt_number - 1} previous attempts):
- Failed paths to avoid: {list(self.global_failed_paths)[-5:] if self.global_failed_paths else 'None'}
- Previously rejected files: {[f['file'] for f in self.global_rejected_files[-3:]] if self.global_rejected_files else 'None'}
- Failed decisions: {len(self.global_failed_decisions)} navigation failures recorded

IMPORTANT: Avoid repeating these failed decisions and paths!
"""
                # Insert the failure context before the task description
                base_prompt = base_prompt.replace(
                    "TASK: Analyze the directories", 
                    failure_context + "\nTASK: Analyze the directories"
                )
            
            return base_prompt
        
        finder.create_navigation_prompt = enhanced_navigation_prompt
        
        # Enhance decision validation
        original_is_repeating = finder.is_repeating_failed_action if hasattr(finder, 'is_repeating_failed_action') else lambda x, y: False
        
        def enhanced_is_repeating(decision: Dict[str, Any], context: Dict[str, Any]) -> bool:
            """Enhanced repetition check including global history."""
            # Check current attempt
            local_repeating = original_is_repeating(decision, context)
            
            # Check global history
            global_repeating = self.should_avoid_decision(decision, context)
            
            return local_repeating or global_repeating
        
        finder.is_repeating_failed_action = enhanced_is_repeating
        
        # Enhance alternative suggestions
        original_suggest = finder.suggest_alternative_action if hasattr(finder, 'suggest_alternative_action') else lambda x, y: x
        
        def enhanced_suggest_alternative(decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
            """Enhanced alternative suggestion using global history."""
            # First try local suggestion
            local_alternative = original_suggest(decision, context)
            
            # Then try global history-based suggestion
            global_alternative = self.suggest_alternative_from_history(decision, context)
            
            # Return the global one if it's different and more confident
            if (global_alternative and 
                global_alternative.get('confidence', 0) > local_alternative.get('confidence', 0)):
                return global_alternative
            
            return local_alternative
        
        finder.suggest_alternative_action = enhanced_suggest_alternative
        
        return finder
    
    def find_file_with_retry(self, user_input: str, max_retries: int = 3) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Main method to find file with intelligent retry mechanism."""
        print(f"🔄 Starting AI File Finder with Enhanced Retry System")
        print(f"📊 Max attempts: {max_retries}")
        print(f"=" * 60)
        
        for attempt in range(1, max_retries + 1):
            print(f"\n🎯 ATTEMPT {attempt} of {max_retries}")
            print(f"{'=' * 50}")
            
            if attempt > 1:
                print(f"🧠 Learning from {len(self.global_failed_decisions)} previous failures")
                print(f"🚫 Avoiding {len(self.global_failed_paths)} known failed paths")
                print(f"📋 {len(self.global_rejected_files)} files already rejected")
            
            try:
                # Create enhanced finder for this attempt
                self.finder = self.create_enhanced_finder(attempt)
                
                # Find starting path
                start_path = find_starting_path(user_input)
                print(f"📂 Starting from: {start_path}")
                
                # Attempt to find the file
                result = self.finder.find_file(user_input, start_path)
                
                if result:
                    print(f"\n🎉 SUCCESS on attempt {attempt}!")
                    return result
                else:
                    print(f"\n❌ Attempt {attempt} failed - no file found")
                    
            except Exception as e:
                print(f"\n💥 Attempt {attempt} failed with error: {e}")
                traceback.print_exc()
            
            # Update global failure tracking
            if self.finder:
                self.update_global_failures(self.finder)
                print(f"\n📈 Updated global failure knowledge:")
                print(f"   - Failed paths: {len(self.global_failed_paths)}")
                print(f"   - Failed decisions: {len(self.global_failed_decisions)}")
                print(f"   - Rejected files: {len(self.global_rejected_files)}")
            
            if attempt < max_retries:
                print(f"\n⏰ Preparing for attempt {attempt + 1}...")
                print(f"🔍 Will avoid previously failed approaches")
            else:
                print(f"\n🔚 All {max_retries} attempts exhausted")
        
        return None


def find_file(user_input, project_name = None, open_file = False):
    """Main entry point using the enhanced retry system."""
    print("AI-Powered File Finder with Intelligent Retry")
    print("=" * 50)

    # Use the enhanced retry system
    retry_system = AIFileFinderWithRetry()
    result = retry_system.find_file_with_retry(user_input, max_retries=3)
    
    if result:
        file_location = result[0]
        file_content = result[1]
        print(f"\n🎉 Successfully found file:")
        print(f"📁 Path: {file_location}")
        print(f"📄 Name: {Path(file_location).name}")

        # Option to open externally
        if open_file:
            os.startfile(file_location)
        return file_location, file_content
    else:
        print(f"\n❌ File search failed after all retry attempts")
        print(f"📊 Total failed paths discovered: {len(retry_system.global_failed_paths)}")
        print(f"📊 Total failed decisions: {len(retry_system.global_failed_decisions)}")
        return None, None


if __name__ == "__main__":
    user_input = "I'm looking for a file that converts NUTS2 to E-highway nodes. I am working on the joule model. Which Belgium node is Brussels placed in?"
    user_input = "I'm looking for the Solar PV Renewable load factors in the node FR02?"

    try:
        file_location, file_content = find_file(user_input)
        if file_location:
            print("\n✅ Search completed successfully!")
        else:
            print("\n❌ Search completed without finding target file")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        traceback.print_exc()

    print("🏁 File finder session complete")
