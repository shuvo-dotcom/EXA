# AI-Driven Demand Settings System

This system uses AI (via `roains`) to automatically determine all parameters for demand settings based on user input and context.

## Features

- **AI-Driven Parameter Selection**: All parameters are chosen by AI based on user input and context
- **Dynamic Dictionary Management**: Automatically adds new projects, scenarios, and carriers if suitable options aren't available
- **Flexible Year Selection**: Can handle both year ranges and specific target years
- **Comprehensive Chronology Options**: Supports multiple time resolutions from yearly to quarter-hourly
- **Intelligent Defaults**: Falls back to sensible defaults if AI responses can't be parsed

## File Structure

```
src/demand/
├── create_demand_settings.py          # Main module with AI-driven functions
├── demand_dictionary_manager.py       # Dictionary management utilities
├── example_usage.py                   # Usage examples
├── demand_dictionaries/
│   ├── projects.json                  # Available project names
│   ├── scenarios.json                 # Available scenario names
│   ├── carriers.json                  # Available energy carriers
│   └── chronology.json                # Available time resolution options
```

## Usage

### Basic Usage

```python
from src.demand.create_demand_settings import create_demand_settings

user_input = "I need to analyze renewable energy transition for the UK from 2030 to 2050"
context = "Energy transition analysis focusing on wind and solar deployment"

settings = create_demand_settings(user_input, context)
```

### Adding New Options

```python
from src.demand.create_demand_settings import add_new_project, add_new_scenario, add_new_carrier

# Add new project
add_new_project("My_New_Project")

# Add new scenario
add_new_scenario("Custom_Scenario")

# Add new carrier
add_new_carrier("Ammonia")
```

### Getting Available Options

```python
from src.demand.create_demand_settings import get_available_options

options = get_available_options()
print(options)
```

## AI Prompts

The system uses specific prompts for each parameter:

1. **Project Selection**: Chooses from available projects or suggests new ones
2. **Scenario Selection**: Matches scenarios to project needs
3. **Climate Year**: Extracts specific years from user input (defaults to 2009)
4. **Carriers Selection**: Selects relevant energy carriers based on project scope
5. **Years Selection**: Determines year ranges or specific target years
6. **Chronology Selection**: Chooses appropriate time resolution
7. **District Heating**: Determines if district heating is relevant

## Default Values

- **Climate Year**: 2009 (if not specified)
- **Chronology**: Daily (if AI selection fails)
- **Carriers**: ['Hydrogen', 'Liquids', 'Heat', 'Biofuels', 'Solids', 'Methane', 'Electricity']
- **Years**: list(range(2025, 2046)) (if AI selection fails)

## Return Format

The `create_demand_settings()` function returns a dictionary with:

```python
{
    'project_name': str,
    'scenario': str,
    'district_heating_demand': bool/None,
    'refclimateyear': int,
    'cy': int,
    'carriers': list,
    'years': list,
    'chronology': str
}
```

## Special Cases

- **Years Selection**: If the AI determines more information is needed, it returns "USER_INPUT_REQUIRED"
- **New Options**: Any new projects, scenarios, or carriers are automatically added to the respective JSON files
- **Error Handling**: Robust parsing with fallbacks to prevent system failures

## Next Steps

The `set_boolean_flags()` function will be updated next to also use AI for determining boolean parameters based on user input and context.
