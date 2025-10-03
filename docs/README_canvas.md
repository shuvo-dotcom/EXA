Modelling Canvas
=================

Simple Streamlit-based interactive canvas to draw PLEXOS objects and relationships.

How to run
----------

From the repository root run:

```powershell
python -m pip install -r requirements.txt
streamlit run src/plexos/modelling_canvas/canvas_app.py
```

What it does
------------
- Lets you add nodes with a selected class (colour-coded)
- Create links: pipeline (solid), membership (dashed), cross-sector (dashed offset)
- Export JSON describing nodes/links
- Generate a short textual description suitable to pass to an LLM or an image model

Notes & next steps
------------------
- This is intentionally lightweight — consider a React + Konva canvas for a richer UX.
- Next improvements: selectable/movable nodes on canvas, save/load multiple diagrams, convert diagram to structured DAG using the existing planning code.
