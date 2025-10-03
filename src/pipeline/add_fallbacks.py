#!/usr/bin/env python
"""
Inject `fallback_of` metadata – and stub any missing fallback functions – into
an existing function-registry JSON file.

Usage:
    python update_registry_with_fallbacks.py \
           --registry function_registry_v02.json \
           --fallback-map fallback_map.json
"""
import json, pathlib, sys, textwrap, typing as t

def load_json(path: pathlib.Path) -> t.Any:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        sys.exit(f"❌  Failed to read {path}: {e}")

def ensure_fallback_entry(reg: dict, primary: str, fb_name: str):
    """Add fallback stub if missing; always tag with fallback_of."""
    if fb_name not in reg:
        reg[fb_name] = {
            "module": "ext",
            "description": f"Auto-generated fallback for {primary}.",
            "inputs": [],
            "outputs": [
                {"name": "result", "type": "Any", "semantic": "fallback"}
            ],
            "side_effect": False,
            "categories": ["fallback", "external"],
        }
    # Tag either way
    reg[fb_name]["fallback_of"] = primary

def main(reg_path: pathlib.Path, fb_map_path: pathlib.Path) -> None:
    data = load_json(reg_path)
    # Allow old flat array OR new {"functions": …}
    func_block = data["functions"] if "functions" in data else data
    registry: dict = (
        func_block[0] if isinstance(func_block, list) else func_block
    )

    fb_map = load_json(fb_map_path)

    for primary, chain in fb_map.items():
        if primary not in registry:
            print(f"⚠️  Primary function '{primary}' missing – skipping")
            continue
        for fb_fn in chain:
            ensure_fallback_entry(registry, primary, fb_fn)

    reg_path.write_text(json.dumps(data, indent=4))
    print(f"✅  {reg_path.name} updated with fallback metadata")

if __name__ == "__main__":
    # Direct function call without command line arguments
    # Update these paths as needed
    registry_path = pathlib.Path("config/function_registry.json")  # Update this path
    fallback_map_path = pathlib.Path("config/fallback_map.json")  # Update this path
    
    main(registry_path, fallback_map_path)
