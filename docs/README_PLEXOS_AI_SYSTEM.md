# PLEXOS Parameter AI Decision System

## Overview

This system uses AI to automatically determine all parameters needed for PLEXOS data extraction based on user input and context. It replaces manual parameter setting with intelligent parameter inference.

## Components

### 1. Core System (`plexos_parameter_ai_system.py`)

The main AI-driven system that determines all PLEXOS extraction parameters:

- **PlexosParameterAIDecisionSystem**: Main class that orchestrates parameter determination
- **create_plexos_parameters_from_input()**: Convenience function for quick parameter generation

### 2. PLEXOS Extraction Utilities (`plexos_extraction_utils.py`)

Utility functions for analyzing PLEXOS files and directories:

- **extract_plexos_models_from_xml()**: Parse PLEXOS XML files for model names
- **scan_directory_for_plexos_files()**: Find PLEXOS files in directories
- **extract_models_from_directory()**: Get all models from a directory
- **extract_run_ids_from_directory()**: Extract available run IDs
- **extract_years_from_directory()**: Extract available years
- **analyze_plexos_structure()**: Comprehensive analysis of PLEXOS directory

### 3. Updated Parameter File (`plexos_extration_ai_decision.py`)

Updated to use AI-driven parameter determination instead of hardcoded values.

### 4. Test and Examples (`test_plexos_ai_system.py`)

Comprehensive testing and example usage of the system.

## Parameters Determined by AI

| Parameter | Description | AI Logic |
|-----------|-------------|----------|
| `simulation_phase` | 'LT' or 'ST' | Analyzes user intent for investment vs dispatch |
| `baselocation` | Path to PLEXOS files | Matches user input to available repositories |
| `model_name` | Project name | Uses demand dictionary manager for available projects |
| `extract_plexos_data` | Boolean flag | True by default, False if user wants to skip extraction |
| `run_file_compiler` | Boolean flag | False by default, True if user mentions compiling outputs |
| `model_version` | Specific model version | Matches user input to available models from PLEXOS |
| `temporal_granularity_levels` | Time resolution | Determines from user requirements (yearly, monthly, daily, hourly) |
| `run_ids` | List of run IDs | Extracts from user input and available runs |
| `years` | List of years | Determines from user input and available years |

## Usage Examples

### Basic Usage

```python
from src.ai.plexos_parameter_ai_system import create_plexos_parameters_from_input

# Simple parameter determination
user_input = "Run long-term investment analysis for 2030 and 2040 with hourly data"
context = "European gas network capacity planning"

parameters = create_plexos_parameters_from_input(user_input, context)
print(parameters)
```

### Advanced Usage

```python
from src.ai.plexos_parameter_ai_system import PlexosParameterAIDecisionSystem

# Initialize system
system = PlexosParameterAIDecisionSystem()

# Determine parameters
parameters = system.determine_all_parameters(user_input, context)

# Generate parameter file content
file_content = system.generate_parameter_file_content(parameters)

# Save to file
with open("my_plexos_parameters.py", "w") as f:
    f.write(file_content)
```

### Direct Parameter File Usage

```python
# Updated plexos_extration_ai_decision.py automatically determines parameters
from src.ai.plexos_parameter_ai_system import create_plexos_parameters_from_input

# Define user requirements
user_input = "Run dispatch optimization for DHEM project, v39 database, 2030 focus"
context = "Short-term operational planning with nuclear scenarios"

# Get AI-determined parameters
parameters = create_plexos_parameters_from_input(user_input, context)

# Parameters are automatically extracted and available for use
simulation_phase = parameters['simulation_phase']
baselocation = parameters['baselocation']
# ... etc
```

## AI Prompting Strategy

The system uses structured AI prompts with:

1. **Context Setting**: Clear description of the parameter's purpose
2. **Available Options**: JSON-formatted lists of valid choices
3. **Guidelines**: Specific rules for parameter selection
4. **Validation**: Response parsing and fallback values
5. **Temperature Control**: Low temperature (0.1) for consistent results

## Integration with Existing Systems

### Demand Dictionary Manager
- Retrieves available project names
- Uses project-specific defaults for boolean flags

### Modelling System Functions
- Loads PLEXOS repository locations
- Provides base location options for AI selection

### PLEXOS Extraction Utilities
- Analyzes actual PLEXOS files for available models, runs, and years
- Provides real data for AI decision-making

## Customization and Extension

### Adding New Parameters

1. Add parameter determination method to `PlexosParameterAIDecisionSystem`
2. Update `determine_all_parameters()` to include new parameter
3. Modify `generate_parameter_file_content()` to include in output

### Implementing PLEXOS Integration

The system provides placeholder functions in `plexos_extraction_utils.py` that you need to implement:

1. **extract_plexos_models_from_xml()**: Parse actual PLEXOS XML structure
2. **extract_run_ids_from_directory()**: Analyze PLEXOS run structure
3. **extract_years_from_directory()**: Extract temporal information

### Customizing AI Behavior

Modify the AI context and prompts in each determination method to:
- Add domain-specific knowledge
- Change decision criteria
- Adjust response formats

## Error Handling and Fallbacks

The system includes comprehensive error handling:

- **File Access Errors**: Graceful fallback to default values
- **AI Response Parsing**: JSON parsing with fallbacks
- **Invalid Selections**: Validation against available options
- **Missing Dependencies**: Import error handling

## Testing and Validation

Run the test suite to validate the system:

```bash
python src/ai/test_plexos_ai_system.py
```

The test includes:
- Multiple user input scenarios
- Parameter validation
- Code generation testing
- Interactive parameter determination

## Performance Considerations

- **Caching**: Consider caching PLEXOS file analysis results
- **Batch Processing**: For multiple parameter sets, batch AI calls
- **Model Selection**: Use appropriate AI model based on complexity needs

## Future Enhancements

1. **Learning System**: Track user corrections to improve AI decisions
2. **Multi-language Support**: Support for non-English user inputs
3. **Advanced Validation**: Cross-parameter validation and consistency checks
4. **Real-time Updates**: Dynamic updating of available options
5. **User Preferences**: Learn and apply user-specific preferences
