"""Interactive modelling canvas for PLEXOS-like objects.

Usage: run with `streamlit run src/plexos/modelling_canvas/canvas_app.py` from the repo root.

Features implemented:
- Load class list from `src/plexos/dictionaries/class_descriptions.yaml` and present as dropdown
- Add circle nodes with class-colour, label and X/Y position (0-100 plane)
- Create links between nodes of type: pipeline (solid), membership (dashed), cross-sector (dashed, representing Z-plane)
- Render an SVG diagram live in the app
- Export JSON description of nodes/links and generate a plain-language description suitable as LLM/image-model context

This is intentionally lightweight (pure Streamlit + SVG) to avoid heavy frontend stacks.
"""

from __future__ import annotations

import json
import hashlib
from typing import Dict, List
from pathlib import Path

import streamlit as st
import yaml
import html

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False

# Paths
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLASS_DESC_PATH = ROOT / "src" / "plexos" / "dictionaries" / "class_descriptions.yaml"


def load_classes(path: Path) -> Dict[str, str]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        main = raw.get("main_classes", {})
        secondary = raw.get("secondary_classes", {})
        classes = {}
        # combine main and secondary preserving order
        for k in sorted(map(int, main.keys())) if isinstance(main, dict) else []:
            classes[str(k)] = main.get(str(k))
        for k in sorted(map(int, secondary.keys())) if isinstance(secondary, dict) else []:
            classes[str(k)] = secondary.get(str(k))
        # Fallback: if above didn't work, try simpler mapping
        if not classes:
            classes = {**(main or {}), **(secondary or {})}
        # Return only names as a list mapping id->name
        return {v: v for v in classes.values()} if classes else {v: v for v in list(main.values()) + list(secondary.values())}
    except Exception:
        return {"Node": "Node", "Generator": "Generator", "Line": "Line", "Pipeline": "Pipeline"}


def class_to_color(name: str) -> str:
    # deterministic hash to pastel color
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    # pastel adjustment
    r = (r + 200) // 2
    g = (g + 200) // 2
    b = (b + 200) // 2
    return f"rgb({r},{g},{b})"


def coord_to_canvas(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    # input x,y in 0-100 range
    cx = (x / 100.0) * width
    cy = height - (y / 100.0) * height
    return cx, cy


def render_svg(nodes: List[Dict], edges: List[Dict], width: int = 900, height: int = 600) -> str:
    svg_parts = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']

    # draw grid
    grid_lines = []
    for gx in range(0, 101, 10):
        x = (gx / 100.0) * width
        grid_lines.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{height}" stroke="#eee" stroke-width="1" />')
    for gy in range(0, 101, 10):
        y = height - (gy / 100.0) * height
        grid_lines.append(f'<line x1="0" y1="{y}" x2="{width}" y2="{y}" stroke="#eee" stroke-width="1" />')
    svg_parts.extend(grid_lines)

    # draw edges (behind nodes)
    for e in edges:
        n1 = next((n for n in nodes if n["id"] == e["from"]), None)
        n2 = next((n for n in nodes if n["id"] == e["to"]), None)
        if not n1 or not n2:
            continue
        x1, y1 = coord_to_canvas(n1["x"], n1["y"], width, height)
        x2, y2 = coord_to_canvas(n2["x"], n2["y"], width, height)
        stroke = "#333"
        stroke_width = 3
        dash = ""
        opacity = 0.9
        if e.get("type") == "membership":
            dash = 'stroke-dasharray="6,6"'
            stroke = "#666"
            stroke_width = 2
        elif e.get("type") == "pipeline":
            stroke = "#0b6efd"
            stroke_width = 3
        elif e.get("type") == "cross-sector":
            stroke = "#d62828"
            dash = 'stroke-dasharray="8,4"'
            stroke_width = 3
            opacity = 0.95

        # small offset for cross-sector to visually separate
        if e.get("type") == "cross-sector":
            dx, dy = ( (y2 - y1) * 0.02, (x1 - x2) * 0.02 )
            x1o, y1o, x2o, y2o = x1 + dx, y1 + dy, x2 + dx, y2 + dy
        else:
            x1o, y1o, x2o, y2o = x1, y1, x2, y2

        svg_parts.append(
            f'<line x1="{x1o}" y1="{y1o}" x2="{x2o}" y2="{y2o}" stroke="{stroke}" stroke-width="{stroke_width}" {dash} stroke-linecap="round" opacity="{opacity}" />'
        )

    # draw nodes
    for n in nodes:
        x, y = coord_to_canvas(n["x"], n["y"], width, height)
        r = 18
        color = n.get("color") or class_to_color(n.get("class", "Node"))
        svg_parts.append(f'<g data-id="{n["id"]}">')
        svg_parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" stroke="#222" stroke-width="1.5" />')
        # label
        svg_parts.append(f'<text x="{x}" y="{y + r + 14}" font-size="12" text-anchor="middle" fill="#111">{html.escape(str(n.get("name","")))}</text>')
        # class small label
        svg_parts.append(f'<text x="{x}" y="{y - r - 8}" font-size="11" text-anchor="middle" fill="#222" opacity="0.9">{html.escape(str(n.get("class","")))}</text>')
        svg_parts.append('</g>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def main():
    # In a multipage app, set_page_config may have already been called
    try:
        st.set_page_config(page_title="Modelling Canvas", layout="wide")
    except Exception:
        pass
    st.title("Modelling Canvas — draw objects, pipelines and memberships")

    classes = load_classes(CLASS_DESC_PATH)
    class_names = sorted(list(classes.keys())) if classes else ["Node", "Generator", "Line"]

    if "nodes" not in st.session_state:
        st.session_state.nodes = []
    if "edges" not in st.session_state:
        st.session_state.edges = []

    mode = st.sidebar.selectbox("Mode", ["Form (legacy)", "Place nodes (click)", "Select / Connect (click)"])
    # initialise helper state
    if "connect_selection" not in st.session_state:
        st.session_state.connect_selection = []
    if "canvas_obj_count" not in st.session_state:
        st.session_state.canvas_obj_count = 0

    left, right = st.columns([1, 2])

    with left:
        st.header("Create node")
        sel_class = st.selectbox("Class", class_names, index=class_names.index("Node") if "Node" in class_names else 0)
        name = st.text_input("Name", value=f"{sel_class}_1")
        col = class_to_color(sel_class)

        if mode == "Form (legacy)":
            x = st.slider("X (0-100)", 0.0, 100.0, 50.0)
            y = st.slider("Y (0-100)", 0.0, 100.0, 50.0)
            if st.button("Add node"):
                nid = f"n{len(st.session_state.nodes) + 1}"
                st.session_state.nodes.append({"id": nid, "name": name or nid, "class": sel_class, "x": x, "y": y, "color": col})
        else:
            if not HAS_CANVAS:
                st.warning("Interactive click-to-place/select requires `streamlit-drawable-canvas`. Install with: pip install streamlit-drawable-canvas")
            else:
                st.info("Click on the canvas (right) to place nodes or select nodes for connecting, depending on the selected mode.")

        st.markdown("---")
        st.header("Create link / pipeline")
        if st.session_state.nodes:
            node_options = {n["id"]: f"{n['name']} ({n['class']})" for n in st.session_state.nodes}
            if mode == "Form (legacy)":
                from_id = st.selectbox("From", options=list(node_options.keys()), format_func=lambda k: node_options[k])
                to_id = st.selectbox("To", options=[k for k in node_options.keys() if k != from_id], format_func=lambda k: node_options[k])
                link_type = st.selectbox("Type", options=["pipeline", "membership", "cross-sector"])
                if st.button("Add link"):
                    st.session_state.edges.append({"from": from_id, "to": to_id, "type": link_type})
            else:
                st.info("Canvas connect mode: click two nodes to select them (they will appear below). Then choose a link type and press 'Create link'.")
                st.write("Selected for connect:", st.session_state.connect_selection)
                link_type = st.selectbox("Type", options=["pipeline", "membership", "cross-sector"], key="connect_type")
                if st.button("Create link"):
                    sel = st.session_state.get('connect_selection', [])
                    if len(sel) >= 2:
                        a, b = sel[0], sel[1]
                        st.session_state.edges.append({"from": a, "to": b, "type": link_type})
                        st.session_state.connect_selection = []
                    else:
                        st.warning("Select two nodes first by clicking the canvas.")

        st.markdown("---")
        st.header("Manage")
        if st.session_state.nodes:
            for n in st.session_state.nodes:
                cols = st.columns([1, 4, 1])
                cols[0].write(f"{n['id']}")
                cols[1].write(f"**{n['name']}** — {n['class']} (@{n['x']:.0f}, {n['y']:.0f})")
                if cols[2].button("Remove", key=f"rm_{n['id']}"):
                    st.session_state.nodes = [x for x in st.session_state.nodes if x["id"] != n["id"]]
                    # also remove edges referencing it
                    st.session_state.edges = [e for e in st.session_state.edges if e["from"] != n["id"] and e["to"] != n["id"]]
                    # Safe rerun: use experimental_rerun when available, otherwise stop and rely on next interaction to refresh
                    try:
                        st.experimental_rerun()
                    except Exception:
                        st.stop()

        if st.button("Clear diagram"):
            st.session_state.nodes = []
            st.session_state.edges = []

        st.markdown("---")
        st.header("Export & describe")
        if st.button("Export JSON"):
            out = {"nodes": st.session_state.nodes, "edges": st.session_state.edges}
            out_path = ROOT / "output" / "canvas_export.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            st.success(f"Exported to {out_path}")

        if st.button("Generate textual description for LLM"):
            desc_lines = []
            desc_lines.append("Diagram description:")
            for n in st.session_state.nodes:
                desc_lines.append(f"Node '{n['name']}' (class={n['class']}) at (x={n['x']:.1f}, y={n['y']:.1f}).")
            for e in st.session_state.edges:
                n1 = next((x for x in st.session_state.nodes if x['id'] == e['from']), None)
                n2 = next((x for x in st.session_state.nodes if x['id'] == e['to']), None)
                if n1 and n2:
                    desc_lines.append(f"{e['type'].capitalize()} link between '{n1['name']}' and '{n2['name']}'.")
            st.code("\n".join(desc_lines))

    with right:
        st.header("Canvas")
        width_px = 900
        height_px = 600
        if HAS_CANVAS and mode != "Form (legacy)":
            canvas_result = st_canvas(
                fill_color="",
                stroke_width=2,
                stroke_color="#000",
                background_color="#fff",
                height=height_px,
                width=width_px,
                drawing_mode="point",
                key="main_canvas",
            )

            if canvas_result is not None and getattr(canvas_result, 'json_data', None):
                objs = canvas_result.json_data.get('objects', [])
                prev_count = st.session_state.get('canvas_obj_count', 0)
                if len(objs) > prev_count:
                    last = objs[-1]
                    left = last.get('left', 0)
                    top = last.get('top', 0)
                    x_pct = (left / width_px) * 100.0
                    y_pct = (1.0 - (top / height_px)) * 100.0
                    if mode == "Place nodes (click)":
                        nid = f"n{len(st.session_state.nodes) + 1}"
                        st.session_state.nodes.append({"id": nid, "name": name or nid, "class": sel_class, "x": x_pct, "y": y_pct, "color": col})
                    elif mode == "Select / Connect (click)":
                        # pick nearest node
                        best = None
                        best_d = 1e9
                        for n in st.session_state.nodes:
                            dx = n['x'] - x_pct
                            dy = n['y'] - y_pct
                            d = (dx*dx + dy*dy)**0.5
                            if d < best_d:
                                best_d = d
                                best = n
                        if best:
                            sel = st.session_state.get('connect_selection', [])
                            if len(sel) < 2:
                                sel.append(best['id'])
                                st.session_state.connect_selection = sel
                    st.session_state.canvas_obj_count = len(objs)

            svg = render_svg(st.session_state.nodes, st.session_state.edges, width=width_px, height=height_px)
            st.components.v1.html(svg, height=height_px + 20)
        else:
            svg = render_svg(st.session_state.nodes, st.session_state.edges)
            st.components.v1.html(svg, height=650)


if __name__ == "__main__":
    main()
