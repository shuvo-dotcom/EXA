"""
Example Prompts for Multi-Agent System
=======================================
This file contains example prompts that demonstrate how the system routes
tasks to Emil, Lola, and Nova based on the nature of the request.
"""

# ==============================================================================
# EMIL - Technical/Engineering Tasks
# ==============================================================================

EMIL_EXAMPLES = [
    {
        "name": "Model Distillation",
        "prompt": """
        Distill the Joule model to Irish geographic scope and add a 
        nuclear power plant with 1000MW capacity connected to the Dublin node.
        Run a basecase scenario and a nuclear scenario.
        """,
        "expected_tasks": [
            "Filter Joule model to Ireland",
            "Add nuclear generator object",
            "Create grid connection",
            "Run basecase simulation",
            "Run nuclear scenario simulation"
        ]
    },
    {
        "name": "Data File Modification",
        "prompt": """
        Modify the H2/Gas Pipeline/Capacities_H2_Year_FID+PCI+PMI.csv file 
        and set all non-zero values to 999. Save as 
        Pipeline/Capacities_H2_Year_FID+PCI+PMI_UNLIMITED.csv
        """,
        "expected_tasks": [
            "Read CSV file",
            "Update values (non-zero → 999)",
            "Write new CSV file"
        ]
    },
    {
        "name": "Network Modeling",
        "prompt": """
        Create a new transmission line connecting Node A to Node B with 
        500MW capacity and 2% losses. Add it to the basecase scenario.
        """,
        "expected_tasks": [
            "Create line object",
            "Set capacity property",
            "Set loss property",
            "Create node memberships",
            "Add to scenario"
        ]
    },
    {
        "name": "Hydrogen Demand Creation",
        "prompt": """
        Build the tst workbooks for hydrogen demand across all European regions.
        """,
        "expected_tasks": [
            "Create demand categories",
            "Generate regional data",
            "Create workbook structure",
            "Populate with demand values"
        ]
    }
]

# ==============================================================================
# LOLA - Content/Communications Tasks
# ==============================================================================

LOLA_EXAMPLES = [
    {
        "name": "Technical Report",
        "prompt": """
        Write a comprehensive report comparing the Irish basecase model outputs 
        to the nuclear scenario. Include executive summary, methodology, findings, 
        charts, and recommendations. Target audience: technical stakeholders.
        """,
        "expected_tasks": [
            "Coordinate with Emil for simulation data",
            "Analyze differences between scenarios",
            "Write executive summary",
            "Create methodology section",
            "Generate findings and visualizations",
            "Write recommendations"
        ]
    },
    {
        "name": "Blog Post",
        "prompt": """
        Write a blog post about the benefits of renewable energy integration 
        in modern power systems. Include sections on solar, wind, and storage 
        technologies. Target audience: general public. Tone: informative and optimistic.
        """,
        "expected_tasks": [
            "Research renewable energy benefits",
            "Write introduction",
            "Create solar section",
            "Create wind section",
            "Create storage section",
            "Write conclusion"
        ]
    },
    {
        "name": "Social Media Campaign",
        "prompt": """
        Create a social media campaign announcing our new energy modeling 
        capabilities. Include posts for LinkedIn, Twitter, and Facebook.
        Focus on accessibility and innovation.
        """,
        "expected_tasks": [
            "Develop campaign strategy",
            "Write LinkedIn post",
            "Write Twitter thread",
            "Write Facebook post",
            "Create engagement hooks"
        ]
    },
    {
        "name": "Press Release",
        "prompt": """
        Write a press release announcing the completion of the Pan-European 
        hydrogen network study. Highlight key findings and business impact.
        """,
        "expected_tasks": [
            "Coordinate with Emil for key findings",
            "Write headline and subtitle",
            "Write opening paragraph",
            "Include quotes and data",
            "Add boilerplate and contact info"
        ]
    }
]

# ==============================================================================
# NOVA - Administrative/Assistant Tasks
# ==============================================================================

NOVA_EXAMPLES = [
    {
        "name": "Log Analysis",
        "prompt": """
        Read the latest log file from the logs directory and provide a summary 
        of any errors or warnings that occurred in the last 24 hours.
        """,
        "expected_tasks": [
            "Locate latest log file",
            "Parse log entries",
            "Filter errors and warnings",
            "Summarize findings"
        ]
    },
    {
        "name": "File Summary",
        "prompt": """
        Read all configuration files in the config directory and provide 
        a summary of the current system settings.
        """,
        "expected_tasks": [
            "List files in config directory",
            "Read each configuration file",
            "Extract key settings",
            "Create summary report"
        ]
    },
    {
        "name": "Status Report",
        "prompt": """
        Provide a status update on all pipeline progress files from the 
        last week. Include completion rates and any failures.
        """,
        "expected_tasks": [
            "List progress files",
            "Parse progress data",
            "Calculate statistics",
            "Generate status report"
        ]
    },
    {
        "name": "Information Retrieval",
        "prompt": """
        Search through the docs directory and find all documentation related 
        to PLEXOS database operations. Provide a summary and list of files.
        """,
        "expected_tasks": [
            "Search docs directory",
            "Filter PLEXOS-related files",
            "Read relevant sections",
            "Create summary and file list"
        ]
    }
]

# ==============================================================================
# MULTI-AGENT - Complex Workflows
# ==============================================================================

MULTI_AGENT_EXAMPLES = [
    {
        "name": "Complete Study with Report",
        "prompt": """
        Distill the Joule model to Irish scope, add a 1000MW nuclear power plant,
        run basecase and nuclear scenarios, and write a comprehensive comparison 
        report for stakeholders.
        """,
        "expected_routing": {
            "task_1": {
                "assistant": "Emil",
                "description": "Model distillation and scenario execution"
            },
            "task_2": {
                "assistant": "Lola",
                "description": "Comparison report writing",
                "depends_on": "task_1"
            }
        }
    },
    {
        "name": "Data Analysis with Documentation",
        "prompt": """
        Analyze the hydrogen pipeline capacity data, identify bottlenecks,
        and create a summary document with findings and recommendations.
        """,
        "expected_routing": {
            "task_1": {
                "assistant": "Emil",
                "description": "Data analysis and bottleneck identification"
            },
            "task_2": {
                "assistant": "Lola",
                "description": "Summary document creation",
                "depends_on": "task_1"
            }
        }
    },
    {
        "name": "Study with Status Updates",
        "prompt": """
        Run a capacity expansion study for the European network, provide 
        progress updates every hour, and create a final technical report.
        """,
        "expected_routing": {
            "task_1": {
                "assistant": "Emil",
                "description": "Capacity expansion study execution"
            },
            "task_2": {
                "assistant": "Nova",
                "description": "Monitor progress and provide updates",
                "parallel_with": "task_1"
            },
            "task_3": {
                "assistant": "Lola",
                "description": "Final technical report",
                "depends_on": "task_1"
            }
        }
    }
]

# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

def print_examples(category="all"):
    """Print example prompts for testing."""
    
    if category in ["all", "emil"]:
        print("\n" + "="*80)
        print("EMIL - Engineering Tasks Examples")
        print("="*80)
        for ex in EMIL_EXAMPLES:
            print(f"\n{ex['name']}:")
            print(f"  Prompt: {ex['prompt'].strip()}")
            print(f"  Expected Tasks: {', '.join(ex['expected_tasks'])}")
    
    if category in ["all", "lola"]:
        print("\n" + "="*80)
        print("LOLA - Communications Tasks Examples")
        print("="*80)
        for ex in LOLA_EXAMPLES:
            print(f"\n{ex['name']}:")
            print(f"  Prompt: {ex['prompt'].strip()}")
            print(f"  Expected Tasks: {', '.join(ex['expected_tasks'])}")
    
    if category in ["all", "nova"]:
        print("\n" + "="*80)
        print("NOVA - Administrative Tasks Examples")
        print("="*80)
        for ex in NOVA_EXAMPLES:
            print(f"\n{ex['name']}:")
            print(f"  Prompt: {ex['prompt'].strip()}")
            print(f"  Expected Tasks: {', '.join(ex['expected_tasks'])}")
    
    if category in ["all", "multi"]:
        print("\n" + "="*80)
        print("MULTI-AGENT - Complex Workflow Examples")
        print("="*80)
        for ex in MULTI_AGENT_EXAMPLES:
            print(f"\n{ex['name']}:")
            print(f"  Prompt: {ex['prompt'].strip()}")
            print(f"  Expected Routing:")
            for task_id, task_info in ex['expected_routing'].items():
                print(f"    {task_id}: {task_info['assistant']} - {task_info['description']}")
                if 'depends_on' in task_info:
                    print(f"      Depends on: {task_info['depends_on']}")


if __name__ == "__main__":
    import sys
    
    category = sys.argv[1] if len(sys.argv) > 1 else "all"
    print_examples(category)
