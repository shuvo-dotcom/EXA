# Generative Prompt Sheet Builder - System Documentation

## Overview

The **Prompt Sheet Builder** is a generative system that creates hierarchical report structures using LLMs. Unlike traditional template-based systems, this builder dynamically generates the complete report outline from a user's request.

## Key Concept

**The CSV files you provided are EXAMPLES of the data structure, NOT the source data.**

The system uses LLMs to CREATE the prompt sheet structure in the same format as those CSVs, but the content is generated fresh for each request.

## How It Works

### Input
```python
user_request = """
Create a comprehensive report on hydrogen sector expansion 
and electricity system integration for 2030-2050.
"""

context = """
The Joule model is a multi-carrier energy system model covering the EU.
"""

dag_info = {
    "name": "PLEXOS Model DAG",
    "available_queries": ["Generator capacity", "Annual generation"]
}
```

### Process (Bottom-Up Generation)

The system builds the structure hierarchically using LLMs:

#### 1. **Generate Aims** (Step 1/7)
```json
[
  {
    "id": 1,
    "title": "Draft comprehensive report on the Joule model",
    "description": "This aim involves creating a detailed report..."
  }
]
```

#### 2. **Generate Objectives** (Step 2/7)
For each Aim, the LLM generates 3-7 Objectives:
```json
[
  {
    "id": 1,
    "aim_id": 1,
    "title": "Executive Summary",
    "description": "Overview of the report..."
  },
  {
    "id": 2,
    "aim_id": 1,
    "title": "Modelling Methodologies",
    "description": "Quick methodological overview..."
  }
]
```

#### 3. **Generate Tasks** (Step 3/7)
For each Objective, the LLM generates 2-6 Tasks (max 6):
```json
[
  {
    "id": "1.1",
    "task_id": 1,
    "objective_id": 1,
    "title": "Executive Summary",
    "description": "Very short summary of Executive Summary page"
  }
]
```

#### 4. **Generate Sub Tasks** (Step 4/7)
For each Task, the LLM generates 1-5 Sub Tasks:
```json
[
  {
    "id": "1.1.1",
    "sub_task_id": 1,
    "task_section_id": "1.1",
    "task_header": "Executive Summary",
    "title": "Opener for Executive Summary",
    "description": "Write an engaging opening paragraph",
    "input": "Text Guidelines",
    "geographic_level": "EU"
  }
]
```

#### 5. **Generate External Search** (Step 5/7)
For each Sub Task, the LLM determines if external data is needed:
```json
[
  {
    "unique_id": 1,
    "id": "1.1.1",
    "title": "Executive Summary",
    "description": "Latest EU hydrogen policy updates",
    "prompt": "Search for recent EU hydrogen strategy documents",
    "data_source": "Internet",
    "level": "High-level summary"
  }
]
```

**DAG Integration:** If `dag_info` is provided, the LLM can suggest DAG searches:
```json
{
  "data_source": "DAG",
  "prompt": "Extract generator capacity by fuel type for 2030",
  "additional_information": "Use PLEXOS solution file"
}
```

#### 6. **Generate Text Guidelines** (Step 6/7)
For each Sub Task, generate writing guidelines using context/RAG:
```json
{
  "id": "1.1.1",
  "description": "What this section should accomplish",
  "standard": "Provide a high-level overview of the model results",
  "advanced": "Include comparative analysis with historical trends",
  "research_topics": "EU hydrogen policy, sector coupling",
  "research_notes": "Reference the latest EU Hydrogen Strategy 2023"
}
```

**RAG Integration:** The `rag_context` parameter can provide project-specific information from a vector database or document collection.

#### 7. **Generate Default Charts** (Step 7/7)
**Currently a placeholder function** - will be implemented later to:
- Explore PLEXOS solution files
- Determine available data series
- Match charts to report sections
- Generate chart configurations

### Output

A complete JSON structure:
```json
{
  "aims": [
    {
      "id": 1,
      "title": "...",
      "objectives": [
        {
          "id": 1,
          "tasks": [
            {
              "id": "1.1",
              "sub_tasks": [
                {
                  "id": "1.1.1",
                  "title": "...",
                  "description": "..."
                }
              ]
            }
          ]
        }
      ]
    }
  ],
  "external_search": [...],
  "text_guidelines": [...],
  "default_charts": []
}
```

## Usage Example

```python
from src.LOLA.prompt_sheet_builder import PromptSheetBuilder

# Initialize
builder = PromptSheetBuilder(
    base_model="gpt-4",
    dag_info=dag_info  # Optional: DAG information
)

# Build prompt sheet
prompt_sheet = builder.build_complete_prompt_sheet(
    user_request="Create a report on hydrogen sector expansion",
    context="Joule multi-carrier energy model for EU",
    rag_context=""  # Optional: from vector database
)

# Export to JSON
builder.export_to_json("output/prompt_sheet.json")

# Get statistics
stats = builder.get_statistics()
print(f"Generated {stats['sub_tasks']} sub-tasks")
```

## Key Features

### 1. **Fully Generative**
- No pre-defined templates
- Structure adapts to user request
- LLM-driven expansion at each level

### 2. **Context-Aware**
- Uses general project context
- Integrates RAG/vector database information
- Considers available DAG capabilities

### 3. **Hierarchical Control**
- Max 6 tasks per objective (enforced)
- 1-5 sub-tasks per task
- Clear parent-child relationships

### 4. **External Search Integration**
- Determines when internet search is needed
- Identifies opportunities for DAG queries
- Generates specific search prompts

### 5. **Text Guidelines with RAG**
- Project-specific writing guidance
- Standard and advanced level instructions
- Research topics and notes

## Integration Points

### 1. **RAG/Vector Database**
```python
# Retrieve context from vector database
rag_context = vector_db.query(sub_task['title'])

# Use in text guidelines generation
guideline = builder.generate_text_guidelines(
    sub_task, 
    context, 
    rag_context=rag_context
)
```

### 2. **DAG System**
```python
dag_info = {
    "name": "PLEXOS Model DAG",
    "description": "Extract data from PLEXOS solution files",
    "available_queries": [
        "Generator capacity by fuel type",
        "Annual generation by region",
        "Transmission flows",
        "System costs",
        "Emissions by sector"
    ]
}

builder = PromptSheetBuilder(dag_info=dag_info)
```

### 3. **Executor Integration**
The generated prompt sheet JSON can be consumed by your existing executor:
```python
# Generated prompt sheet
prompt_sheet = builder.build_complete_prompt_sheet(...)

# Pass to executor
executor.execute(prompt_sheet)
```

## Next Steps

### 1. **Implement Default Charts Function**
Will require:
- PLEXOS solution file explorer
- Property and data series detector
- Chart configuration generator
- Section-to-chart matcher

### 2. **Add RAG Integration**
- Connect to vector database
- Query relevant documents per sub-task
- Inject context into text guidelines

### 3. **DAG Search Executor**
- Parse DAG search configurations
- Execute DAG queries
- Return structured data
- Integrate results into report

### 4. **Internet Search Executor**
- Parse internet search configurations
- Execute web searches
- Extract relevant information
- Format for report inclusion

## Architecture

```
User Request
     ↓
PromptSheetBuilder
     ↓
[Generate Aims] → LLM Call #1
     ↓
For each Aim:
    [Generate Objectives] → LLM Call #2-4
        ↓
        For each Objective:
            [Generate Tasks] → LLM Call #5-10
                ↓
                For each Task:
                    [Generate Sub Tasks] → LLM Call #11-N
                        ↓
                        For each Sub Task:
                            [Generate External Search] → LLM Call
                            [Generate Text Guidelines] → LLM Call
     ↓
[Generate Default Charts] → Placeholder
     ↓
Complete Prompt Sheet JSON
     ↓
Export to File
     ↓
Feed to Executor
```

## File Structure

```
src/LOLA/
├── prompt_sheet_builder.py          # Main builder class
├── generated_prompt_sheet.json      # Output file
└── README_prompt_sheet_builder.md   # This documentation
```

## CSV Examples (Reference Only)

The CSV files you provided serve as **format examples**:
- `Aims.csv` - Shows aim structure
- `Objectives.csv` - Shows objective structure
- `Tasks.csv` - Shows task structure
- `Sub_Tasks.csv` - Shows sub-task structure
- `External_Search.csv` - Shows search config structure
- `Text_Guidelines.csv` - Shows guideline structure
- `Default_Charts.csv` - Shows chart config structure (to be implemented)

These are **NOT** loaded by the system - they're just reference examples of the output format.

## Benefits

1. **Flexibility**: Adapts to any report request
2. **Consistency**: Uses standard hierarchical structure
3. **Intelligence**: LLM determines relevant sections
4. **Scalability**: Can generate small or large structures
5. **Integration**: Works with RAG, DAG, and external search
6. **Automation**: Minimal manual configuration needed
