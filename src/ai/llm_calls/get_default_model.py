import os
import yaml

def load_default_model(config_path: str = 'config/default_ai_models.yaml') -> str:
    """Load the default AI model from a YAML configuration file."""
    default_ai_models_file = r'config\default_ai_models.yaml'
    with open(default_ai_models_file, 'r') as f:
        ai_models_config = yaml.safe_load(f)
    base_model = ai_models_config.get("base_model", "gpt-5-mini")
    pro_model = ai_models_config.get("pro_model", "gpt-5")

    return base_model, pro_model