#!/usr/bin/env python3
"""
AI-Powered File Finder
An intelligent file finder that uses LLM to navigate directories and find the right files
based on natural language user input.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from src.tools.simple_file_browser import SimpleFileBrowser
from src.ai.llm_calls.llm_integration import LLMManager
import src.ai.file_operations.ai_finder_config as config

class AIFileFinder:
    """AI-powered file finder using LLM for intelligent navigation."""
    
    def __init__(self, api_key_file: str = None):
        """Initialize the AI file finder."""
        self.browser = SimpleFileBrowser()
        self.conversation_history = []
        self.current_path = None
        self.search_context = ""
        self.found_files = []
        self.llm_manager = LLMManager()
        
        print(f"🤖 AI File Finder initialized")
        if self.llm_manager.current_provider:
            print(f"🔗 Connected to {self.llm_manager.current_provider}")
        else:
            print(f"⚠️ Running in simulation mode (no API keys found)")

    def make_llm_call(self, prompt: str, system_prompt: str = None) -> str:
        """Make a call to the LLM API."""
        try:
            if config.SHOW_LLM_REASONING:
                print(f"🤖 Making LLM call...")
            
            response = self.llm_manager.make_call(prompt, system_prompt)
            
            if config.VERBOSE_LOGGING:
                print(f"💭 LLM Response received")
            
            return response
        except Exception as e:
            print(f"❌ LLM API call failed: {e}")
            return self.fallback_decision(prompt)

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

    def create_navigation_prompt(self, user_query: str, context: Dict[str, Any]) -> str:
        """Create prompt for LLM to decide navigation."""
        system_prompt = config.SYSTEM_PROMPTS["navigation"]
        
        prompt = f"""
USER QUERY: "{user_query}"

CURRENT CONTEXT:
- Current path: {context.get('current_path', 'Unknown')}
- Available subdirectories: {context.get('subdirectories', [])}
- Total files in current directory: {context.get('file_count', 0)}
- Supported files visible: {context.get('supported_files', [])}
- Total supported files: {context.get('total_supported_files', 0)}

CONVERSATION HISTORY:
{self.format_conversation_history()}

TASK: Analyze the directories and files available and decide the next step.

Respond with a JSON object containing:
{{
    "action": "navigate|examine|back|complete",
    "reasoning": "Explain your decision process", 
    "choice": "directory_name|files|back|done",
    "confidence": 0.0-1.0
}}

Actions:
- "navigate": Choose a subdirectory to explore (choice = directory name)
- "examine": Look at files in current directory (choice = "files")
- "back": Go back to parent directory (choice = "back") 
- "complete": Found what we're looking for (choice = "done")
"""
        return prompt

    def create_file_analysis_prompt(self, user_query: str, files: List[str]) -> str:
        """Create prompt for LLM to analyze files."""
        system_prompt = config.SYSTEM_PROMPTS["file_analysis"]
        
        prompt = f"""
USER QUERY: "{user_query}"

AVAILABLE FILES:
{self.format_file_list(files)}

CONVERSATION HISTORY:
{self.format_conversation_history()}

TASK: Choose which file to examine first based on the user's requirements.

Respond with a JSON object:
{{
    "action": "examine",
    "reasoning": "Why you chose this file",
    "choice": file_number,
    "confidence": 0.0-1.0
}}

Choose the file number (1-{len(files)}) that seems most relevant to the user's query.
"""
        return prompt

    def create_file_evaluation_prompt(self, user_query: str, file_path: str, file_content: Dict[str, Any]) -> str:
        """Create prompt for LLM to evaluate if file is correct."""
        system_prompt = config.SYSTEM_PROMPTS["content_evaluation"]
        content_summary = self.summarize_file_content(file_content)
        
        prompt = f"""
USER QUERY: "{user_query}"

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
        """Format conversation history for LLM context."""
        if not self.conversation_history:
            return "No previous steps taken."
        
        history = []
        for i, step in enumerate(self.conversation_history[-5:], 1):  # Last 5 steps
            history.append(f"{i}. {step}")
        
        return "\n".join(history)

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

    def find_file(self, user_query: str, start_path: str = None) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Main method to find file based on user query."""
        print(f"🔍 AI File Finder")
        print(f"=" * 50)
        print(f"Query: {user_query}")
        print(f"=" * 50)
        
        self.search_context = user_query
        self.current_path = start_path or os.getcwd()
        self.conversation_history = []
        
        max_steps = config.MAX_SEARCH_STEPS
        step_count = 0
        
        while step_count < max_steps:
            step_count += 1
            print(f"\n🔄 Step {step_count}")
            
            # Get current directory context
            context = self.get_directory_context(self.current_path)
            
            if context.get('error'):
                print(f"❌ Error accessing directory: {context['error']}")
                return None
            
            print(f"📂 Current: {self.current_path}")
            print(f"📁 Subdirectories: {len(context['subdirectories'])}")
            print(f"📄 Files: {context['total_supported_files']}")
            
            # Check if we have files to examine
            if context['total_supported_files'] > 0:
                # Ask LLM whether to examine files or continue navigating
                nav_prompt = self.create_navigation_prompt(user_query, context)
                nav_response = self.make_llm_call(nav_prompt)
                
                try:
                    nav_decision = json.loads(nav_response)
                    print(f"🤖 LLM Decision: {nav_decision['action']}")
                    print(f"💭 Reasoning: {nav_decision['reasoning']}")
                    
                    if nav_decision['action'] == 'examine':
                        # Examine files in current directory
                        all_files = self.browser.search_folder(self.current_path)
                        supported_files = self.browser.filter_supported_files(all_files)
                        
                        if supported_files:
                            # Ask LLM to choose which file to examine
                            file_prompt = self.create_file_analysis_prompt(user_query, supported_files)
                            file_response = self.make_llm_call(file_prompt)
                            
                            try:
                                file_decision = json.loads(file_response)
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
                                        eval_prompt = self.create_file_evaluation_prompt(user_query, selected_file, file_content)
                                        eval_response = self.make_llm_call(eval_prompt)
                                        
                                        try:
                                            eval_decision = json.loads(eval_response)
                                            print(f"\n🤖 Evaluation: {eval_decision['reasoning']}")
                                            
                                            if eval_decision.get('is_correct', False):
                                                print(f"✅ Found the correct file!")
                                                return selected_file, file_content
                                            else:
                                                print(f"❌ Not the right file. {eval_decision.get('next_step', 'Continuing search...')}")
                                                self.log_step(f"File {Path(selected_file).name} not correct - {eval_decision['reasoning']}")
                                                
                                                # Ask user if they want to continue or manually override
                                                user_input = input("\nContinue search? (y/n/manual): ").strip().lower()
                                                if user_input == 'n':
                                                    return None
                                                elif user_input == 'manual':
                                                    return self.manual_file_selection(supported_files)
                                                
                                        except json.JSONDecodeError:
                                            print("❌ Error parsing LLM evaluation response")
                                            return None
                                            
                            except json.JSONDecodeError:
                                print("❌ Error parsing LLM file selection response")
                                return None
                    
                    elif nav_decision['action'] == 'navigate':
                        # Navigate to chosen directory
                        choice = nav_decision.get('choice', '')
                        subdirectories = self.browser.get_subdirectories(self.current_path)
                        
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
                            return None
                    
                    elif nav_decision['action'] == 'back':
                        # Go back to parent directory
                        parent_path = Path(self.current_path).parent
                        if str(parent_path) != self.current_path:
                            self.current_path = str(parent_path)
                            self.log_step(f"Went back to: {Path(parent_path).name}")
                        else:
                            print("❌ Already at root directory")
                            return None
                    
                    elif nav_decision['action'] == 'complete':
                        print("✅ Search completed by LLM decision")
                        return None
                        
                except json.JSONDecodeError:
                    print("❌ Error parsing LLM navigation response")
                    return None
            
            else:
                # No files in current directory, navigate to subdirectories
                subdirectories = self.browser.get_subdirectories(self.current_path)
                if subdirectories:
                    nav_prompt = self.create_navigation_prompt(user_query, context)
                    nav_response = self.make_llm_call(nav_prompt)
                    
                    try:
                        nav_decision = json.loads(nav_response)
                        choice = nav_decision.get('choice', '')
                        
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
                            return None
                            
                    except json.JSONDecodeError:
                        print("❌ Error parsing LLM response")
                        return None
                else:
                    print("❌ No subdirectories or files found")
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
                    
            except ValueError:
                print("Please enter a valid number")


def main():
    """Main entry point."""
    print("AI-Powered File Finder")
    print("=" * 30)
    
    # Get user query
    user_query = input("What file are you looking for? Describe it: ").strip()
    if not user_query:
        print("No query provided. Exiting.")
        return
    
    # Get starting path
    start_path = input("Starting directory (Enter for current): ").strip()
    if not start_path:
        start_path = os.getcwd()
    
    # Initialize and run AI file finder
    finder = AIFileFinder()
    result = finder.find_file(user_query, start_path)
    
    if result:
        file_path, file_content = result
        print(f"\n🎉 Successfully found file:")
        print(f"📁 Path: {file_path}")
        print(f"📄 Name: {Path(file_path).name}")
        
        # Option to open externally
        choice = input("\nOpen file externally? (y/n): ").strip().lower()
        if choice == 'y':
            finder.browser.open_file_externally(file_path)
    else:
        print("\n❌ File search was not successful")


if __name__ == "__main__":
    main()
