#!/usr/bin/env python
"""Export the LangGraph topology as Mermaid source and PNG images.

Writes into ``docs/``:

* ``langgraph_structure_raw.mmd`` / ``langgraph_structure.png`` — generated
  straight from the compiled graph, so they can never drift from
  ``app.graph.builder``.
* ``langgraph_structure.mmd`` / ``langgraph_structure_annotated.png`` — the
  hand-labelled version. LangGraph's exporter cannot name the branches a
  conditional edge takes, and the review loops are the whole point of this
  graph, so the router literals are written on the edges here.

Rendering goes through mermaid.ink (the same service ``draw_mermaid_png``
uses), so it needs network access.

    python scripts/export_graph_image.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.runnables.graph_mermaid import _render_mermaid_using_api

from app.graph.builder import build_graph_definition

DOCS = Path(__file__).resolve().parents[1] / "docs"


def render(mermaid: str, png_path: Path) -> None:
    png_path.write_bytes(
        _render_mermaid_using_api(mermaid, background_color="white", file_type="png")
    )
    print(f"wrote {png_path.relative_to(DOCS.parent)}")


def main() -> None:
    DOCS.mkdir(exist_ok=True)

    raw = build_graph_definition().compile().get_graph().draw_mermaid()
    (DOCS / "langgraph_structure_raw.mmd").write_text(raw)
    print("wrote docs/langgraph_structure_raw.mmd")
    render(raw, DOCS / "langgraph_structure.png")

    annotated = DOCS / "langgraph_structure.mmd"
    if annotated.exists():
        render(annotated.read_text(), DOCS / "langgraph_structure_annotated.png")
    else:
        print(f"skipped annotated render: {annotated} is missing")


if __name__ == "__main__":
    main()
