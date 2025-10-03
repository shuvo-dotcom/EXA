"""
EMIL - Director of Engineering
===============================
Technical operations, model management, and data processing.
"""

# Lazy import to avoid circular dependency with pipeline_executor_v05
def main(*args, **kwargs):
    from src.EMIL.Emil import main as emil_main
    return emil_main(*args, **kwargs)

__all__ = ['main']
