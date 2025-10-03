#!/usr/bin/env python3
"""
LLM Integration Module for AI File Finder
Provides actual API integrations for OpenAI, Anthropic, and other providers.
"""

import os
import json
import requests
from typing import Dict, Any, Optional


class LLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def make_request(self, prompt: str, system_prompt: str = None) -> str:
        """Make a request to the LLM API."""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        super().__init__(api_key)
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def make_request(self, prompt: str, system_prompt: str = None) -> str:
        """Make request to OpenAI API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        super().__init__(api_key)
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"
    
    def make_request(self, prompt: str, system_prompt: str = None) -> str:
        """Make request to Anthropic API."""
        try:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": self.model,
                "max_tokens": 500,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            if system_prompt:
                data["system"] = system_prompt
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["content"][0]["text"]
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        super().__init__(api_key)
        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    def make_request(self, prompt: str, system_prompt: str = None) -> str:
        """Make request to Gemini API."""
        try:
            headers = {"Content-Type": "application/json"}
            
            content = prompt
            if system_prompt:
                content = f"{system_prompt}\n\n{prompt}"
            
            data = {
                "contents": [{
                    "parts": [{"text": content}]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 500
                }
            }
            
            params = {"key": self.api_key}
            
            response = requests.post(self.base_url, headers=headers, json=data, params=params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
            
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")


class LLMManager:
    """Manages different LLM providers and handles API calls."""
    
    def __init__(self):
        self.providers = {}
        self.current_provider = None
        self.load_providers()
    
    def load_providers(self):
        """Load available API providers based on available keys."""
        # Try to load OpenAI
        openai_key = self.get_api_key(["api_keys/openai", "api_keys/open_ai"], "OPENAI_API_KEY")
        if openai_key:
            self.providers["openai"] = OpenAIProvider(openai_key)
            print("✅ OpenAI provider loaded")
        
        # Try to load Anthropic
        anthropic_key = self.get_api_key(["api_keys/claude"], "ANTHROPIC_API_KEY")
        if anthropic_key:
            self.providers["anthropic"] = AnthropicProvider(anthropic_key)
            print("✅ Anthropic provider loaded")
        
        # Try to load Gemini
        gemini_key = self.get_api_key(["api_keys/gemini"], "GOOGLE_API_KEY")
        if gemini_key:
            self.providers["gemini"] = GeminiProvider(gemini_key)
            print("✅ Gemini provider loaded")
        
        # Set default provider
        if "openai" in self.providers:
            self.current_provider = "openai"
        elif "anthropic" in self.providers:
            self.current_provider = "anthropic"
        elif "gemini" in self.providers:
            self.current_provider = "gemini"
        
        if self.current_provider:
            print(f"🤖 Using {self.current_provider} as default provider")
        else:
            print("⚠️ No LLM providers available - using simulation mode")
    
    def get_api_key(self, file_paths: list, env_var: str) -> Optional[str]:
        """Get API key from file or environment variable."""
        # Try files first
        for path in file_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        key = f.read().strip()
                        if key:
                            return key
                except Exception:
                    continue
        
        # Try environment variable
        return os.environ.get(env_var)
    
    def make_call(self, prompt: str, system_prompt: str = None, provider: str = None) -> str:
        """Make LLM API call using specified or default provider."""
        if not self.providers:
            return self.simulation_response(prompt)
        
        provider_name = provider or self.current_provider
        if provider_name not in self.providers:
            return self.simulation_response(prompt)
        
        try:
            provider = self.providers[provider_name]
            response = provider.make_request(prompt, system_prompt)
            return response
        except Exception as e:
            print(f"❌ LLM API call failed: {e}")
            return self.simulation_response(prompt)
    
    def simulation_response(self, prompt: str) -> str:
        """Provide simulated responses when no API is available."""
        prompt_lower = prompt.lower()
        
        if "analyze directories" in prompt_lower or "current context" in prompt_lower:
            if "src" in prompt_lower:
                return json.dumps({
                    "action": "navigate",
                    "reasoning": "The 'src' directory typically contains source code files, which might be what we're looking for.",
                    "choice": "src",
                    "confidence": 0.8
                })
            elif "docs" in prompt_lower:
                return json.dumps({
                    "action": "navigate", 
                    "reasoning": "Documentation directory might contain the files we need.",
                    "choice": "docs",
                    "confidence": 0.7
                })
            else:
                return json.dumps({
                    "action": "examine",
                    "reasoning": "Let me examine the files in the current directory first.",
                    "choice": "files",
                    "confidence": 0.6
                })
        
        elif "available files" in prompt_lower:
            return json.dumps({
                "action": "examine",
                "reasoning": "I'll examine the first file to see if it matches the requirements.",
                "choice": 1,
                "confidence": 0.6
            })
        
        elif "file being evaluated" in prompt_lower:
            return json.dumps({
                "action": "continue",
                "reasoning": "This appears to be a simulation response. In a real scenario, I would analyze the file content.",
                "is_correct": False,
                "confidence": 0.3,
                "next_step": "Continue searching for a more relevant file"
            })
        
        else:
            return json.dumps({
                "action": "manual",
                "reasoning": "Simulation mode - manual intervention needed",
                "choice": "user_input",
                "confidence": 0.0
            })
    
    def get_available_providers(self) -> list:
        """Get list of available providers."""
        return list(self.providers.keys())
    
    def switch_provider(self, provider_name: str) -> bool:
        """Switch to a different provider."""
        if provider_name in self.providers:
            self.current_provider = provider_name
            print(f"🔄 Switched to {provider_name} provider")
            return True
        else:
            print(f"❌ Provider {provider_name} not available")
            return False


# Example usage and testing
def test_llm_integration():
    """Test LLM integration with available providers."""
    print("Testing LLM Integration...")
    
    manager = LLMManager()
    
    if not manager.providers:
        print("No API providers available - testing simulation mode")
    
    test_prompt = """
    You are helping find a Python file that contains data processing functions.
    
    Current directory contains:
    - src/ (subdirectory)
    - data/ (subdirectory) 
    - main.py (file)
    - config.json (file)
    
    What should be the next step?
    
    Respond with JSON: {"action": "navigate|examine", "choice": "directory_name|files", "reasoning": "explanation"}
    """
    
    response = manager.make_call(test_prompt)
    print(f"Response: {response}")


if __name__ == "__main__":
    test_llm_integration()
