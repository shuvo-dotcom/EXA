r"""AI Meeting Assistant
=================================
Utility to:
1. Enumerate a root Meetings directory (projects at level 1).
2. Build an index of CSV files (lightweight metadata + small content samples).
3. Given a natural language prompt, select relevant CSV files using a hybrid heuristic + LLM (run_open_ai_ns) approach.
4. (Optionally) Send selected CSV content to the LLM for answering the user prompt.

NOTE: This script cannot (from the hosted assistant) list directories outside the current workspace. You must run it locally
to produce the actual project/subfolder listing for: C:\\Users\\ENTSOE\\Tera-joule\\Terajoule - Terajoule\\Meetings

Example CLI usage (PowerShell):
  python -m src.NOVA.ai_meeting_assistant --root "C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Meetings" --build-index
  python -m src.NOVA.ai_meeting_assistant --root "C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Meetings" --query "Summarize last week's turbine maintenance discussions"

Design choices:

Extend easily by swapping model or enriching metadata.
"""

from __future__ import annotations


import csv
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
	# Import the provided LLM helper
	from src.ai.open_ai_calls import run_open_ai_ns as roains  # type: ignore
except Exception:  # pragma: no cover
	roains = None  # Fallback; user must ensure dependency available when running.

DEFAULT_MODEL = "gpt-4.1"
INDEX_FILENAME = "meeting_index.json"
SAMPLE_BYTES = 2048
SAMPLE_MAX_LINES = 40
MAX_CONTENT_BYTES_PER_FILE_FOR_ANSWER = 8000
MAX_FILES_FOR_ANSWER = 5


@dataclass
class CSVFileMeta:
	project: str
	relative_path: str  # relative to root
	file_name: str
	size_bytes: int
	modified_ts: float
	header: List[str]
	sample_rows: List[List[str]]
	sample_text: str

	def to_public_dict(self) -> Dict[str, Any]:
		d = asdict(self)
		# reduce size for context by excluding raw sample_rows (optional)
		return {
			"project": d["project"],
			"relative_path": d["relative_path"],
			"file_name": d["file_name"],
			"size_bytes": d["size_bytes"],
			"modified_iso": datetime.fromtimestamp(self.modified_ts).isoformat(timespec="seconds"),
			"header": self.header,
			"sample_rows": self.sample_rows[:3],  # only a few rows
			"sample_text_excerpt": self.sample_text[:400],
		}


def iter_csv_files(root: Path):
	for p in root.rglob("*.csv"):
		if p.is_file():
			yield p


def safe_read_csv_sample(path: Path) -> Tuple[List[str], List[List[str]], str]:
	header: List[str] = []
	rows: List[List[str]] = []
	sample_text = ""
	try:
		# Raw text sample
		with path.open("r", encoding="utf-8", errors="ignore") as f:
			raw = f.read(SAMPLE_BYTES)
			sample_text = raw
		# Structured sample
		with path.open("r", encoding="utf-8", errors="ignore") as f:
			reader = csv.reader(f)
			header = next(reader, [])
			for i, row in enumerate(reader):
				if i >= min(5, SAMPLE_MAX_LINES):
					break
				rows.append(row)
	except Exception:
		pass
	return header, rows, sample_text


def build_index(root: Path) -> List[CSVFileMeta]:
	root = root.expanduser().resolve()
	records: List[CSVFileMeta] = []
	for csv_path in iter_csv_files(root):
		try:
			rel = csv_path.relative_to(root).as_posix()
			project = rel.split("/", 1)[0]  # level 1 directory
			stat = csv_path.stat()
			header, sample_rows, sample_text = safe_read_csv_sample(csv_path)
			records.append(
				CSVFileMeta(
					project=project,
					relative_path=rel,
					file_name=csv_path.name,
					size_bytes=stat.st_size,
					modified_ts=stat.st_mtime,
					header=header,
					sample_rows=sample_rows,
					sample_text=sample_text,
				)
			)
		except Exception:
			continue
	return records


def save_index(records: List[CSVFileMeta], out_path: Path):
	payload = [asdict(r) for r in records]
	out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_index(path: Path) -> List[CSVFileMeta]:
	data = json.loads(path.read_text(encoding="utf-8"))
	records: List[CSVFileMeta] = []
	for d in data:
		records.append(CSVFileMeta(**d))
	return records


def heuristic_rank(records: List[CSVFileMeta], query: str) -> List[Tuple[CSVFileMeta, float]]:
	terms = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", query) if len(t) > 2]
	scored: List[Tuple[CSVFileMeta, float]] = []
	for r in records:
		text_blob = " ".join([
			r.relative_path.lower(),
			" ".join(r.header).lower(),
			r.sample_text.lower(),
		])
		score = sum(text_blob.count(term) * 2 for term in terms)
		# mild size penalty to prefer smaller files
		score -= r.size_bytes / 1_000_000.0
		scored.append((r, score))
	scored.sort(key=lambda x: x[1], reverse=True)
	return scored


LLM_SELECTION_SYSTEM = """You are a file selection assistant. You receive a user query and a catalog of CSV files.
Return a concise JSON object with keys: chosen_files (list of relative_path) and reasoning (short string). Only choose files clearly relevant. Prefer <=5 files. If nothing fits, return an empty list. Do NOT hallucinate paths."""


def llm_choose_files(query: str, candidate_records: List[CSVFileMeta], model: str = DEFAULT_MODEL) -> Dict[str, Any]:
	if roains is None:
		raise RuntimeError("roains (run_open_ai_ns) import failed; ensure src.ai.open_ai_calls is available.")
	catalog_snippets = [rec.to_public_dict() for rec in candidate_records[:20]]  # limit context
	user_prompt = json.dumps({"query": query, "catalog": catalog_snippets}, ensure_ascii=False)
	raw = roains(user_prompt, LLM_SELECTION_SYSTEM, model=model)
	# Attempt to extract JSON
	json_match = re.search(r"\{[\s\S]*\}", raw)
	if json_match:
		try:
			return json.loads(json_match.group(0))
		except json.JSONDecodeError:
			pass
	# Fallback
	return {"chosen_files": [], "reasoning": "Failed to parse LLM response", "raw": raw}


def read_files_for_answer(root: Path, relative_paths: List[str]) -> Dict[str, str]:
	contents: Dict[str, str] = {}
	for rel in relative_paths[:MAX_FILES_FOR_ANSWER]:
		p = root / rel
		if not p.exists():
			continue
		try:
			with p.open("r", encoding="utf-8", errors="ignore") as f:
				contents[rel] = f.read(MAX_CONTENT_BYTES_PER_FILE_FOR_ANSWER)
		except Exception:
			continue
	return contents


FINAL_ANSWER_SYSTEM = """You are an analytical assistant. You are given (file_path -> partial CSV content excerpts) and a user query.
Provide a concise, factual answer referencing file names when helpful. If data insufficient, state what is missing."""


def llm_answer(query: str, file_contents: Dict[str, str], model: str = DEFAULT_MODEL) -> str:
	if roains is None:
		raise RuntimeError("roains (run_open_ai_ns) import failed; ensure src.ai.open_ai_calls is available.")
	user_payload = json.dumps({"query": query, "files": file_contents}, ensure_ascii=False)
	return roains(user_payload, FINAL_ANSWER_SYSTEM, model=model)


class MeetingFileAgent:
	def __init__(self, root: Path, model: str = DEFAULT_MODEL):
		self.root = root.expanduser().resolve()
		self.model = model
		self.records: List[CSVFileMeta] = []

	# --- Index operations -------------------------------------------------
	def build_index(self):
		self.records = build_index(self.root)
		return self.records

	def load_index_from_file(self, path: Optional[Path] = None):
		path = path or (Path(__file__).parent / INDEX_FILENAME)
		if path.exists():
			self.records = load_index(path)
		return self.records

	def save_index_to_file(self, path: Optional[Path] = None):
		path = path or (Path(__file__).parent / INDEX_FILENAME)
		save_index(self.records, path)
		return path

	# --- Query flow -------------------------------------------------------
	def select_files(self, query: str) -> Dict[str, Any]:
		if not self.records:
			raise RuntimeError("Index empty. Build or load index first.")
		ranked = heuristic_rank(self.records, query)
		top_candidates = [r for r, _ in ranked[:30]]
		llm_result = {}
		try:
			llm_result = llm_choose_files(query, top_candidates, model=self.model)
		except Exception as e:
			llm_result = {"chosen_files": [], "reasoning": f"LLM selection failed: {e}"}

		chosen = llm_result.get("chosen_files", [])
		if not chosen:
			# fallback: top 3 heuristics
			chosen = [r.relative_path for r, _ in ranked[:3]]
			llm_result["fallback"] = True
			llm_result["chosen_files"] = chosen
		return llm_result

	def answer(self, query: str) -> Dict[str, Any]:
		selection = self.select_files(query)
		files = selection.get("chosen_files", [])
		file_contents = read_files_for_answer(self.root, files)
		answer_text = ""
		error = None
		try:
			answer_text = llm_answer(query, file_contents, model=self.model)
		except Exception as e:
			error = f"Answer generation failed: {e}"
		return {
			"query": query,
			"selection": selection,
			"answer": answer_text,
			"error": error,
			"used_files": list(file_contents.keys()),
		}


def print_project_structure(records: List[CSVFileMeta]):
	by_project: Dict[str, List[CSVFileMeta]] = {}
	for r in records:
		by_project.setdefault(r.project, []).append(r)
	for project, items in sorted(by_project.items()):
		print(f"Project: {project}  (CSV files: {len(items)})")
		subfolders = sorted({Path(r.relative_path).parent.as_posix() for r in items})
		for sf in subfolders:
			print(f"  - {sf}")
		print()


def main():  # pragma: no cover
	# Direct function call without command line arguments
	# You can modify these parameters as needed
	root = Path("path/to/your/meetings/directory")  # Update this path
	model = DEFAULT_MODEL
	
	agent = MeetingFileAgent(root, model=model)
	
	index_path = Path(__file__).parent / INDEX_FILENAME
	
	# Build index if it doesn't exist
	if not index_path.exists():
		print(f"Building index from {root} ...")
		records = agent.build_index()
		agent.save_index_to_file(index_path)
		print(f"Indexed {len(records)} CSV files -> {index_path}")
	else:
		# Load existing index
		agent.load_index_from_file(index_path)
		print(f"Loaded {len(agent.records)} records from {index_path}")
	
	# Example usage - you can modify this as needed
	# print_project_structure(agent.records)
	
	# Example query - you can modify this as needed
	# query = "Your query here"
	# result = agent.answer(query)
	# print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
	try:
		main()
	except KeyboardInterrupt:
		print("Interrupted")
	except Exception as e:
		print("Error:", e)
		traceback.print_exc()

