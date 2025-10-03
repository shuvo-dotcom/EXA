import ast
import json
import os
import sys
from typing import Any, Dict, List, Optional

def parse_functions(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse Python file to extract function definitions with metadata.
    Returns a list of dicts with keys: name, module, description, inputs, outputs
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source)
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            doc = ast.get_docstring(node) or ""
            # inputs
            inputs = []
            for arg in node.args.args:
                name = arg.arg
                if arg.annotation:
                    dtype = ast.unparse(arg.annotation)
                else:
                    dtype = "Any"
                inputs.append({"name": name, "dtype": dtype})
            # outputs
            if node.returns:
                dtype = ast.unparse(node.returns)
            else:
                dtype = "Any"
            outputs = [{"dtype": dtype}]
            functions.append({
                "name": func_name,
                "module": module_name,
                "description": doc,
                "inputs": inputs,
                "outputs": outputs
            })
    return functions

def update_registry(registry_path: str, new_entries: List[Dict[str, Any]]) -> None:
    """
    Load existing registry, append new entries, and write back.
    """
    if not os.path.isfile(registry_path):
        print(f"Registry file not found: {registry_path}")
        return
    with open(registry_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    funcs = data.get('functions', [])
    existing_names = {f.get('name') for f in funcs}
    for entry in new_entries:
        if entry['name'] not in existing_names:
            funcs.append(entry)
        else:
            # Optionally update existing entry
            print(f"Skipping existing function: {entry['name']}")
    data['functions'] = funcs
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Registry updated at {registry_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python update_function_registry.py <python_file> <function_registry.json>")
        sys.exit(1)
    python_file = sys.argv[1]
    registry_file = sys.argv[2]
    new_funcs = parse_functions(python_file)
    update_registry(registry_file, new_funcs)

if __name__ == '__main__':
    main()
