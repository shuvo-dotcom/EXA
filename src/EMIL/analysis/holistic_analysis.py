"""Holistic PLEXOS Model Analysis Agent
=======================================

Implements the user-defined pipeline to:
1. Let an LLM choose a project (database folder) from `config/file_locations.json`.
2. Choose a PLEXOS model XML file within that project folder.
3. Open the DB (copy) via `load_plexos_xml`.
4. Retrieve model objects (class "Model").
5. List available solution / related files in the folder.
6. Choose countries (LLM) from user prompt producing name + ISO2.
7. Get active classes from DB.
8. Choose relevant classes (LLM) from active list.
9. For each (country, class) extract basic data and run an LLM analysis.

NOTE: Generic per-class detailed property extraction can be expanded later; currently we
extract object names for each class and pass them for reasoning. This provides a scaffold
that can be deepened with property-level pulls.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

import pandas as pd

try:
	# Local imports (guarded so the module can be introspected without full env)
	from src.ai.llm_calls.open_ai_calls import run_open_ai_ns as roains
	from src.EMIL.plexos.plexos_extraction_functions_agents import (
		load_plexos_xml,
		get_active_classes,
		get_objects,
	)
except Exception as e:  # pragma: no cover - import errors surfaced at runtime
	raise

PROJECT_FILE = Path("config/file_locations.json")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ----------------------------- Utility Functions ----------------------------- #

def _read_projects() -> Dict[str, Dict[str, str]]:
	if not PROJECT_FILE.exists():
		raise FileNotFoundError(f"Project file not found: {PROJECT_FILE}")
	with PROJECT_FILE.open("r", encoding="utf-8") as f:
		return json.load(f)


def _llm_json(prompt: str, sys_context: str, model: str = None) -> Any:
	"""Call LLM and attempt to parse JSON response robustly."""
	resp = roains(prompt, sys_context) if model is None else roains(prompt, sys_context, model=model)
	if not resp:
		return None
	# Attempt direct parse
	try:
		return json.loads(resp)
	except Exception:
		# Try to extract JSON block
		import re

		match = re.search(r"```json\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", resp, re.IGNORECASE)
		if match:
			try:
				return json.loads(match.group(1))
			except Exception:
				pass
		# Last resort: find first brace
		brace_idx = resp.find("{")
		if brace_idx != -1:
			tail = resp[brace_idx:]
			try:
				return json.loads(tail)
			except Exception:
				return {"raw": resp}
		return {"raw": resp}


def choose_project(user_input: str) -> Tuple[str, Path]:
	projects = _read_projects()
	listing = [f"{name}: {meta.get('description','')} -> {meta.get('location','')}" for name, meta in projects.items()]
	prompt = f"""
	The user wants to perform a holistic multi-country, multi-class analysis on a PLEXOS model.
	Available projects (name: description -> path):\n""" + "\n".join(listing) + "\n\nReturn JSON: {\n  'chosen_project': <exact name>,\n  'reason': <short>\n}"
	resp = _llm_json(prompt, "You pick the most relevant energy system project.")
	if not resp or 'chosen_project' not in resp:
		# Fallback pick first
		chosen = next(iter(projects.keys()))
	else:
		chosen = resp['chosen_project'] if resp['chosen_project'] in projects else next(iter(projects.keys()))
	path = Path(projects[chosen]['location'])
	return chosen, path


def choose_model_xml(project_path: Path) -> Path:
	xml_files = list(project_path.rglob("*.xml"))
	if not xml_files:
		raise FileNotFoundError(f"No XML models found under {project_path}")
	if len(xml_files) == 1:
		return xml_files[0]
	listing = [str(p) for p in xml_files[:30]]  # limit
	prompt = "List of candidate PLEXOS XML model files:\n" + "\n".join(listing) + "\nReturn JSON {'chosen_xml': <exact path from list>}"
	resp = _llm_json(prompt, "Select the most suitable base model.")
	if resp and 'chosen_xml' in resp and resp['chosen_xml'] in listing:
		return Path(resp['chosen_xml'])
	return xml_files[0]


def list_solution_files(project_path: Path) -> List[str]:
	patterns = ("*.csv", "*.zip", "*.mdb", "*.db", "*.sqlite", "*.parquet")
	sol_files = []
	for pat in patterns:
		sol_files.extend([str(p) for p in project_path.glob(pat)])
	return sol_files


def choose_countries(user_prompt: str) -> List[Dict[str, str]]:
	country_prompt = f"""
	User analysis intent: {user_prompt}
	Provide a focused list (3-12) of most relevant countries (primarily Europe, but include others if clearly relevant).
	Return strict JSON array of objects: [{{'country':'Germany','code':'DE'}}, ...]
	Use ISO 3166-1 alpha-2 uppercase codes. No commentary.
	"""
	resp = _llm_json(country_prompt, "You output only valid JSON with country list; keep to relevance.")
	if isinstance(resp, list):
		# Basic validation
		clean = []
		for item in resp:
			if isinstance(item, dict) and 'country' in item and 'code' in item and len(item['code']) == 2:
				clean.append({'country': item['country'], 'code': item['code'].upper()})
		if clean:
			return clean
	# Fallback minimal set
	return [
		{"country": "Germany", "code": "DE"},
		{"country": "France", "code": "FR"},
		{"country": "Spain", "code": "ES"},
	]


def choose_relevant_classes(user_prompt: str, active_classes: List[str]) -> List[str]:
	prompt = f"""
	User wants: {user_prompt}
	Active PLEXOS classes: {active_classes}
	Pick only those classes most relevant to deliver the requested analysis. Return JSON: {{'classes': ['Class1','Class2', ...]}}.
	Ensure all selections come from the provided active list; limit 3-12 unless user explicitly requests ALL. If request clearly broad (e.g. 'holistic', 'all classes'), return all.
	"""
	resp = _llm_json(prompt, "Select subset of classes.")
	if resp and isinstance(resp, dict) and 'classes' in resp:
		chosen = [c for c in resp['classes'] if c in active_classes]
		if chosen:
			return chosen
	# Fallback: first 5
	return active_classes[:5]



def analyze_subset(user_prompt: str, country: Dict[str, str], class_name: str, subset_df: pd.DataFrame) -> str:
	context = f"You are an energy modelling analysis agent. Provide concise analytical insight (<=120 words) for class '{class_name}' and country {country['country']} ({country['code']}). Use ONLY provided JSON data; if sparse, state limitations succinctly."  # noqa: E501
	payload = subset_df.to_dict(orient="records")[:200]  # cap size
	prompt = (
		f"User prompt: {user_prompt}\n\nInput Data JSON:\n" + json.dumps(payload, ensure_ascii=False) +
		"\nReturn a short analysis paragraph."
	)
	resp = roains(prompt, context)
	return resp or "No analysis returned"


def run_holistic_analysis(user_prompt: str, save: bool = True) -> Dict[str, Any]:
	# 1 + 2 Choose project & model
	project_name, project_path = choose_project(user_prompt)
	model_xml = choose_model_xml(project_path)

	# 3 Open DB
	db = load_plexos_xml(str(model_xml), new_copy=True)

	# 4 Get model objects
	model_objects = get_model_objects(db)

	# 5 Solution files
	solution_files = list_solution_files(project_path)

	# 6 Countries
	countries = choose_countries(user_prompt)

	# 7 Active classes
	active = get_active_classes(db)

	# 8 Choose relevant classes
	chosen_classes = choose_relevant_classes(user_prompt, active)

	# 9 Loop & analyze
	analyses: Dict[str, Dict[str, str]] = {}
	for country in countries:
		ccode = country['code']
		country_dict: Dict[str, str] = {}
		for cls in chosen_classes:
			df_cls = extract_class_objects(db, cls)
			if not df_cls.empty:
				subset = df_cls[df_cls['country_code_inferred'] == ccode]
			else:
				subset = df_cls
			analysis_text = analyze_subset(user_prompt, country, cls, subset)
			country_dict[cls] = analysis_text
		analyses[ccode] = {
			'country_name': country['country'],
			'analyses': country_dict
		}

	result = {
		'timestamp': datetime.utcnow().isoformat() + 'Z',
		'user_prompt': user_prompt,
		'project': project_name,
		'project_path': str(project_path),
		'model_xml': str(model_xml),
		'model_objects_count': len(model_objects),
		'solution_files': solution_files,
		'countries': countries,
		'active_classes_count': len(active),
		'chosen_classes': chosen_classes,
		'analyses': analyses,
	}

	if save:
		out_file = OUTPUT_DIR / f"holistic_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
		with out_file.open('w', encoding='utf-8') as f:
			json.dump(result, f, indent=2, ensure_ascii=False)
		result['output_file'] = str(out_file)
	return result


def main():  # pragma: no cover - simple CLI
	res = run_holistic_analysis(prompt)
	print(json.dumps(res, indent=2)[:4000])


if __name__ == "__main__":  # pragma: no cover
	main()

