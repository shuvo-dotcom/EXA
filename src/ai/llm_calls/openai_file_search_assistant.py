# -*- coding: utf-8 -*-
"""
Created: 2025-09-02
Updated: Handle File Search supported types and attach CSV/XLS to Code Interpreter
Author: Terajoule / ENTSOE

What this script does
---------------------
• Creates a vector store and indexes ONLY File Search–supported files (PDF/TXT/MD/HTML/JSON/Docx/PPTX/code, etc.)
• Uploads tabular files (CSV/XLS/XLSX) to the Assistants file store and attaches them to the user message for Code Interpreter
• Creates an Assistant with the right tools (file_search and/or code_interpreter), runs a Q&A, and returns text + citations

Usage
-----
Update the `files` list and `question` in the __main__ block and run.
"""

import os
import yaml
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from get_api_keys import get_api_key


# ---------------------------
# Config & Client
# ---------------------------
API_KEY = get_api_key('openai')
if API_KEY:
    os.environ['OPENAI_API_KEY'] = API_KEY
else:
    print("Failed to load OpenAI API key.")

DEFAULT_MODELS_YAML = r'config\default_ai_models.yaml'
if os.path.exists(DEFAULT_MODELS_YAML):
    with open(DEFAULT_MODELS_YAML, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    DEFAULT_MODEL = cfg.get("open_api_assistants", "gpt-4o")
else:
    DEFAULT_MODEL = "gpt-4o"

client = OpenAI()

# ---------------------------
# Supported file sets
# ---------------------------
SUPPORTED_FOR_FILE_SEARCH = {
    ".pdf", ".txt", ".md", ".html",
    ".doc", ".docx", ".ppt", ".pptx",
    ".json",
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".rb", ".php", ".tex", ".css"
}

TABULAR_FOR_CODE_INTERPRETER = {".csv", ".xlsx", ".xls"}


def _partition_files(file_paths: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Split files into:
      docs -> sent to File Search
      tabular -> attached to Code Interpreter (NOT indexed)
      unsupported -> skipped entirely
    """
    docs, tabular, unsupported = [], [], []
    for p in file_paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in SUPPORTED_FOR_FILE_SEARCH:
            docs.append(p)
        elif ext in TABULAR_FOR_CODE_INTERPRETER:
            tabular.append(p)
        else:
            unsupported.append(p)
    return docs, tabular, unsupported


def _open_streams(paths: List[str]):
    streams = []
    for p in paths:
        streams.append(open(p, "rb"))
    return streams


# ---------------------------
# Main helper
# ---------------------------
def ask_with_file_search(
                            question: str,
                            file_paths: List[str],
                            assistant_name: str = "Docs+Data Assistant",
                            instructions: str = (
                                "You are a retrieval + analysis assistant. "
                                "Use File Search for documents and Code Interpreter for any data or calculations. "
                                "Cite sources with brief quotes when helpful."
                            ),
                            model: str = None,
                        ) -> Dict[str, Any]:
    """
    Orchestrates File Search (vector store) + Code Interpreter in one run:
    1) Partition files by capability
    2) Create vector store & upload supported docs (if any)
    3) Upload tabular files and prepare code_interpreter attachments (if any)
    4) Create Assistant with proper tools
    5) Create thread, send message (with attachments), run & poll
    6) Return answer text + citations + bookkeeping
    """
    model = model or DEFAULT_MODEL

    # --- 1) Partition
    docs, tabular, unsupported = _partition_files(file_paths)

    # --- 2) Vector store for docs (if any)
    vector_store_id = None
    if docs:
        vs = client.beta.vector_stores.create(name=f"{assistant_name} Vector Store")
        vector_store_id = vs.id
        doc_streams = _open_streams(docs)
        try:
            client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store_id,
                files=doc_streams,
            )
        finally:
            for s in doc_streams:
                try:
                    s.close()
                except Exception:
                    pass

    # --- 3) Upload tabular files to general file storage for Code Interpreter
    attachments = []
    tabular_ids = []
    for p in tabular:
        f = client.files.create(file=open(p, "rb"), purpose="assistants")
        tabular_ids.append(f.id)
        attachments.append({"file_id": f.id, "tools": [{"type": "code_interpreter"}]})

    # --- 4) Create Assistant with appropriate tools
    tools = []
    tool_resources = {}
    # Always include Code Interpreter because user might compute even without tabular files
    tools.append({"type": "code_interpreter"})
    if vector_store_id:
        tools.append({"type": "file_search"})
        tool_resources["file_search"] = {"vector_store_ids": [vector_store_id]}

    assistant = client.beta.assistants.create(
        name=assistant_name,
        instructions=instructions,
        model=model,
        tools=tools,
        tool_resources=tool_resources or None,
    )

    # --- 5) Thread + Message (attach tabular files for CI)
    thread = client.beta.threads.create()
    if attachments:
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=question,
            attachments=attachments,
        )
    else:
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=question,
        )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
        # Optional tuning:
        # max_prompt_tokens=24000,
        # additional_instructions="Prefer concise citations."
    )

    # --- 6) Collect answer + citations
    msgs = client.beta.threads.messages.list(thread_id=thread.id)
    answer_text = ""
    citations = []
    if msgs.data:
        # newest first
        msg = msgs.data[0]
        for item in msg.content:
            if getattr(item, "type", None) == "text":
                answer_text += item.text.value
                for ann in (item.text.annotations or []):
                    if getattr(ann, "type", None) == "file_citation":
                        file_id = ann.file_citation.file_id
                        quote = getattr(ann.file_citation, "quote", "")
                        try:
                            meta = client.files.retrieve(file_id)
                            filename = getattr(meta, "filename", file_id)
                        except Exception:
                            filename = file_id
                        citations.append({"filename": filename, "quote": quote})

    return {
        "status": run.status,
        "assistant_id": assistant.id,
        "thread_id": thread.id,
        "vector_store_id": vector_store_id,
        "files_indexed": docs,
        "files_tabular": tabular,
        "files_unsupported": unsupported,
        "answer": (answer_text or "").strip(),
        "citations": citations,
    }


# ---------------------------
# (Optional) CSV patcher kept for convenience
# ---------------------------
def modify_data_file(
    plexos_file_path: str,
    data_file_name: str,
    user_input: str,
    output_file_name: str,
    test_mode: bool = False,
) -> Dict[str, Any]:
    """
    Your original Code Interpreter helper to modify a CSV with pandas.
    """
    if test_mode:
        print("Running in test mode. No actual file modification will occur.")
        return {
            "status": "success",
            "file_path": r'H2\Gas Pipeline\Capacities_H2_Year_FID+PCI+PMI_Unlimited.csv',
        }

    local_csv_path = os.path.join(plexos_file_path, data_file_name)
    output_file_path = os.path.join(plexos_file_path, output_file_name)

    # 1) upload the CSV to assistants files
    up = client.files.create(file=open(local_csv_path, "rb"), purpose="assistants")

    # 2) create assistant (CI only)
    assistant = client.beta.assistants.create(
        name="CSV Patcher",
        instructions="You are a data-wrangling assistant. Write Python with pandas and save modified files.",
        model=DEFAULT_MODEL,
        tools=[{"type": "code_interpreter"}],
    )

    # 3) create thread + user message with CI attachment
    th = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=th.id,
        role="user",
        content=user_input,
        attachments=[{"file_id": up.id, "tools": [{"type": "code_interpreter"}]}],
    )

    # 4) run & poll
    run = client.beta.threads.runs.create_and_poll(thread_id=th.id, assistant_id=assistant.id)

    # 5) fetch messages and try to download the first returned file (if any)
    msgs = client.beta.threads.messages.list(thread_id=th.id).data
    if not msgs:
        return {"status": run.status, "file_path": None}

    assistant_msg = msgs[0]  # newest first
    returned_file_id = None

    # Try to pull file_ids from attachments (if present)
    for att in getattr(assistant_msg, "attachments", []) or []:
        if hasattr(att, "file_id"):
            returned_file_id = att.file_id
            break

    if returned_file_id:
        content = client.files.content(returned_file_id)
        with open(output_file_path, "wb") as f:
            f.write(content.read())
        return {"status": "success", "file_path": output_file_path}

    return {"status": run.status, "file_path": None}


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Mix of documents (indexed) and CSV (CI attachment)
    files = [
                r"C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Sectoral Model\Nodes and Grid\NUTS Regions\NUTS2 Ehighway Mapping.csv"
            ]
    question = """
                    Which Joule nodes should be used to create terminals for my model considering the cities from the search results below, if not available return the most appropriate node?
                    European marine import terminals for kerosene and methanol, identifying major hubs (Rotterdam, Amsterdam, Antwerp‑Bruges, Hamburg, Marseille;
                    Rotterdam, Amsterdam, Rostock, Huelva) and reporting available capacity estimates (Rotterdam ~7.5M m3 total terminals; Amsterdam >1.3M m3; Vopak Dupeg ~700k m3; Euroports Rostock 700k m3). 
                    Noted growing methanol projects (GIDARA 90kt/yr, Huelva 300kt/yr, Lowlands 120kt, Green2x 320kt) and seasonal kerosene import peaks (May–Aug); precise per‑terminal seasonal capacities 
                    largely unavailable.
                    You are the end of the line so you must return a joule node, if you dont know guess, if you cant guess return a the node with the closest geographical location.
                """

    result = ask_with_file_search(
        question,
        files,
        assistant_name="Demand Profiles RAG",
        instructions=(
            "You are a retrieval + analysis assistant. "
            "Use File Search to cite the methodology and definitions; "
            "use Code Interpreter to read and analyze any attached CSV/XLS files. "
            "Keep answers concise, include short citations."
        ),
    )

    print("\n=== STATUS ===", result["status"])
    print("Indexed (File Search):", result["files_indexed"])
    print("Tabular (Code Interpreter attachments):", result["files_tabular"])
    print("Unsupported (skipped):", result["files_unsupported"])
    print("\n=== ANSWER ===\n", result["answer"])
    if result["citations"]:
        print("\n=== CITATIONS ===")
        for i, c in enumerate(result["citations"], 1):
            print(f"[{i}] {c['filename']} — “{c['quote']}”")

    # Example: keep your CSV patcher available
    # patched = modify_data_file(
    #     plexos_file_path=r"C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\DHEM\National Trends\2030\H2\Gas Pipeline",
    #     data_file_name="Capacities_H2_FIDPCIPMIADVANCED.csv",
    #     user_input="Replace every non-zero number in the value column with 999 and return the new CSV.",
    #     output_file_name="Capacities_H2_FIDPCIPMIADVANCED_UNLIMITED.csv",
    # )
    # print("\nCSV patch result:", patched)
