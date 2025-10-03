# Boolean Flags System Documentation

## Overview

The boolean flags system uses AI to determine various processing options for demand analysis based on user input, context, and project-specific requirements.

## Boolean Flags

### 1. terajoule_framework
- **Purpose**: Determines if the terajoule framework should be used for energy units
- **Default**: `True` (for all projects)
- **AI Logic**: Set to `False` only if user explicitly mentions not using terajoule or wanting different energy units
- **Example Triggers**: "don't use terajoule", "use different energy units"

### 2. extract_heat
- **Purpose**: Removes heat demand from other energy carrier profiles when heat is modelled separately
- **Default**: `False` (for all projects)
- **AI Logic**: Set to `True` when user indicates heat will be modelled in heat class separately
- **Example Triggers**: 
  - "create electricity profiles but remove heat demand as it will be modelled directly in the heat class"
  - "extract heat demand separately"
  - "model heat in heat class"

### 3. extract_transport
- **Purpose**: Removes transport demand from other energy carrier profiles when transport is modelled separately
- **Default**: `False` (for all projects)
- **AI Logic**: Set to `True` when user indicates transport will be modelled in transport class separately
- **Example Triggers**: 
  - "create electricity profiles but remove transport demand as it will be modelled directly in the transport class"
  - "extract transport demand separately"
  - "model transport in transport class"

### 4. extract_hybrid_heating
- **Purpose**: Processes hybrid heating systems (heat pumps with backup heating)
- **Default**: `False` (for all projects)
- **AI Logic**: Set to `True` when user mentions hybrid heating systems
- **Example Triggers**: "hybrid heating", "heat pumps with backup heating", "dual heating systems"

### 5. run_energy_carrier_swapping
- **Purpose**: Swaps energy carriers from one to another (specific sensitivity analysis)
- **Default**: 
  - `True` for Joule_Model
  - `False` for all other projects
- **AI Logic**: Set to `True` for Joule_Model or when user specifically mentions energy carrier swapping
- **Example Triggers**: "swap hydrogen to electricity", "convert methane demand to hydrogen"

### 6. aggregate_sectors
- **Purpose**: Combines all sectors (residential, tertiary, industry, transport) into single demand per carrier
- **Default**: 
  - `False` for Joule_Model
  - `True` for TYNDP_2026_Scenarios, Core_Flexibility_Report, and others
- **AI Logic**: Project-specific defaults unless user specifies otherwise
- **Example Triggers (for False)**: "keep sectors separate", "decompose by sector", "sector-specific analysis"

### 7. interpolate_demand
- **Purpose**: Creates demand profiles for missing years between target years
- **Default**: 
  - `True` for Core_Flexibility_Report
  - `False` for all other projects
- **AI Logic**: Project-specific defaults unless user specifies otherwise
- **Example Triggers (for True)**: "interpolate between years", "fill missing years", "create continuous timeline"

### 8. create_sub_nodes
- **Purpose**: Decomposes countries into finer granularity topology
- **Default**: 
  - `True` for Joule_Model
  - `False` for all other projects
- **AI Logic**: Project-specific defaults unless user specifies otherwise
- **Example Triggers (for True)**: "sub-nodes", "regional breakdown", "decompose countries", "finer granularity"

## Project-Specific Defaults

### Joule_Model
- `run_energy_carrier_swapping`: True
- `aggregate_sectors`: False
- `create_sub_nodes`: True

### Core_Flexibility_Report
- `interpolate_demand`: True

### TYNDP_2026_Scenarios
- `aggregate_sectors`: True

## AI Prompt Structure

Each flag uses a structured prompt that includes:

1. **Context**: Project name, user input, and context
2. **Description**: What the flag does
3. **Default Logic**: Project-specific or general defaults
4. **Trigger Examples**: Specific phrases that should trigger True/False
5. **Instructions**: Clear guidance on when to set True/False

## Usage Examples

### Example 1: Heat Extraction
```python
user_input = "Create electricity profiles but remove heat demand as it will be modelled directly in the heat class"
context = "Heat separation analysis"
project_name = "Core_Flexibility_Report"

# Result: extract_heat = True
```

### Example 2: Joule Model Regional Analysis
```python
user_input = "Regional analysis for European energy system"
context = "Detailed regional breakdown"
project_name = "Joule_Model"

# Result: create_sub_nodes = True, aggregate_sectors = False, run_energy_carrier_swapping = True
```

### Example 3: Transport Extraction
```python
user_input = "Create hydrogen profiles but remove transport demand as it will be modelled directly in the transport class"
context = "Separate transport modelling"
project_name = "TYNDP_2026_Scenarios"

# Result: extract_transport = True
```

## Testing

The system includes comprehensive testing functions:

- `test_boolean_flags()`: Tests various scenarios
- `compare_flag_implementations()`: Compares different implementations
- `get_flag_explanations()`: Provides detailed explanations

## Configuration

Boolean flag configurations are stored in `demand_dictionaries/boolean_flags.json` and can be easily modified to add new triggers or change defaults.

## Error Handling

The system includes robust error handling:
- Fallback to project-specific defaults if AI parsing fails
- Clear logging of AI responses
- Validation of boolean responses
