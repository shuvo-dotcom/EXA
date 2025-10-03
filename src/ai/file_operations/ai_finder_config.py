# AI File Finder Configuration

# API Configuration
DEFAULT_API_PROVIDER = "openai"  # Options: openai, anthropic, gemini, local
MAX_SEARCH_STEPS = 25
CONFIDENCE_THRESHOLD = 0.7

# File Analysis Settings
MAX_FILE_PREVIEW_LENGTH = 500
MAX_FILES_TO_SHOW = 10
MAX_CONVERSATION_HISTORY = 5

# Supported File Extensions (inherited from SimpleFileBrowser)
ADDITIONAL_EXTENSIONS = [
    '.yml', '.yaml',
    '.ini', '.cfg', '.conf',
    '.log', '.md'
]

# LLM Prompt Templates
SYSTEM_PROMPTS = {
    "navigation": """You are an expert file system navigator helping users find specific files. 
    Analyze directory structures and make intelligent decisions about where to look next.
    Always provide JSON responses with clear reasoning.""",
    
    "file_analysis": """You are analyzing files to determine if they match user requirements.
    Consider file names, types, sizes, and content when making decisions.
    Provide clear reasoning for your choices.""",
    
    "content_evaluation": """You are evaluating file contents to determine if they match what the user is looking for.
    Consider the user's original query and the file's actual content.
    Be decisive but explain your reasoning clearly."""
}

# Fallback Behavior
FALLBACK_TO_MANUAL = True
SHOW_LLM_REASONING = True
VERBOSE_LOGGING = True
