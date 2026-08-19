"""Generate docs/API_Reference.md from live module docstrings.

Run via:
    make docs
or directly:
    python tools/generate_api_md.py
"""

from __future__ import annotations

import importlib
import inspect
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Docstring parser (NumPy style) ────────────────────────────────────────────

def _parse_numpy_docstring(doc: str) -> dict:
    result = {"summary": "", "params": [], "returns": "", "returns_type": "", "attributes": [], "raises": []}
    if not doc:
        return result

    lines = doc.strip().splitlines()
    summary_lines, i = [], 0

    while i < len(lines):
        if i + 1 < len(lines) and set(lines[i + 1].strip()) <= {"-"} and lines[i + 1].strip():
            break
        summary_lines.append(lines[i].strip())
        i += 1
    result["summary"] = " ".join(ln for ln in summary_lines if ln).strip()

    sections: dict[str, list[str]] = {}
    while i < len(lines):
        header = lines[i].strip()
        if i + 1 < len(lines) and set(lines[i + 1].strip()) <= {"-"} and lines[i + 1].strip():
            sec: list[str] = []
            i += 2
            while i < len(lines):
                if i + 1 < len(lines) and set(lines[i + 1].strip()) <= {"-"} and lines[i + 1].strip():
                    break
                sec.append(lines[i])
                i += 1
            sections[header.lower()] = sec
        else:
            i += 1

    def _dedent(block: list[str]) -> list[str]:
        return textwrap.dedent("\n".join(block)).splitlines()

    def _parse_entries(raw: list[str]) -> list[tuple[str, str, str]]:
        block = _dedent(raw)
        entries, j = [], 0
        while j < len(block):
            m = re.match(r"^(\*{0,2}\w[\w\s,*]*?)\s*:\s*(.+)$", block[j])
            if m:
                pname, ptype, desc_parts = m.group(1).strip(), m.group(2).strip(), []
                j += 1
                while j < len(block):
                    ln = block[j]
                    if not ln.strip():
                        j += 1
                        continue
                    if ln[0] in (" ", "\t"):
                        desc_parts.append(ln.strip())
                        j += 1
                    else:
                        break
                entries.append((pname, ptype, " ".join(desc_parts)))
            else:
                j += 1
        return entries

    for key, target in [("parameters", "params"), ("attributes", "attributes"), ("raises", "raises")]:
        if key in sections:
            result[target] = _parse_entries(sections[key])

    if "returns" in sections:
        ret = _dedent(sections["returns"])
        type_line = ret[0].strip() if ret else ""
        desc_lines = [ln.strip() for ln in ret[1:] if ln.strip()]
        if type_line and not re.match(r"^\w[\w\s,*]*?\s*:", type_line):
            result["returns_type"] = type_line
            result["returns"] = " ".join(desc_lines)
        else:
            result["returns"] = " ".join(ln.strip() for ln in ret if ln.strip())

    return result


def _best_docstring(obj) -> str:
    if not inspect.isclass(obj):
        return inspect.getdoc(obj) or ""
    class_doc = inspect.getdoc(obj) or ""
    init_doc = inspect.getdoc(obj.__init__) or ""
    if re.search(r"^Parameters\s*\n\s*-{3,}", class_doc, re.MULTILINE):
        return class_doc
    if re.search(r"^Parameters\s*\n\s*-{3,}", init_doc, re.MULTILINE):
        return init_doc
    return class_doc or init_doc


# ── Markdown renderers ────────────────────────────────────────────────────────

def _short_sig(name: str, obj) -> str:
    """Return a readable signature, collapsing to Name(...) when long."""
    try:
        full = str(inspect.signature(obj))
    except (ValueError, TypeError):
        return f"{name}(...)"
    return f"{name}{full}" if len(full) <= 80 else f"{name}(...)"


def _table(entries: list[tuple[str, str, str]], col0: str = "Parameter") -> list[str]:
    """Render (name, type, desc) entries as a markdown table."""
    lines = [
        f"| {col0} | Type | Description |",
        "|-----------|------|-------------|",
    ]
    for pname, ptype, pdesc in entries:
        ptype_md = ptype.replace("|", "\\|")
        pdesc_md = pdesc.replace("|", "\\|") if pdesc else "—"
        lines.append(f"| `{pname}` | `{ptype_md}` | {pdesc_md} |")
    return lines


def _render_item(name: str, obj) -> str:
    kind = "class" if inspect.isclass(obj) else "func"
    parsed = _parse_numpy_docstring(_best_docstring(obj))
    summary = parsed["summary"] or "*No description available.*"

    lines = [
        f"#### `{_short_sig(name, obj)}`",
        f"> **{kind}**",
        "",
        summary,
    ]

    for section_label, key, col0 in [
        ("Parameters", "params", "Parameter"),
        ("Attributes", "attributes", "Attribute"),
        ("Raises", "raises", "Exception"),
    ]:
        entries = parsed.get(key, [])
        if entries:
            lines += ["", f"<details><summary>{section_label}</summary>", ""]
            lines += _table(entries, col0)
            lines.append("</details>")

    rtype = parsed.get("returns_type", "")
    rdesc = parsed.get("returns", "")
    if rtype or rdesc:
        parts = ([f"`{rtype}`"] if rtype else []) + ([rdesc] if rdesc else [])
        lines += ["", f"**Returns:** {' — '.join(parts)}"]

    return "\n".join(lines)


def _render_module(mod, display_name: str, import_path: str, description: str = "") -> str:
    lines = [
        f"### {display_name}",
        f"```python",
        f"import {import_path}",
        f"```",
    ]
    if description:
        lines += ["", description]
    lines += ["", "---"]

    for name in sorted(mod.__all__):
        obj = getattr(mod, name, None)
        if obj is not None:
            lines += ["", _render_item(name, obj), "", "---"]

    return "\n".join(lines)


def _render_section(specs: list, title: str = "") -> str:
    lines = []
    if title:
        lines += [f"## {title}", ""]

    for display_name, import_path, desc in specs:
        lines.append("")
        try:
            mod = importlib.import_module(import_path)
            lines.append(_render_module(mod, display_name, import_path, desc))
        except Exception as exc:
            lines += [f"### {display_name}", f"> ⚠️ Import failed: `{exc}`"]

    return "\n".join(lines)


# ── Module specs (keep in sync with API_Reference.ipynb) ─────────────────────

CORE = [
    (
        "IO — `data_conduit.io`",
        "data_conduit.io",
        "The main entry point for loading data. `collect_dfs` walks a directory tree and "
        "returns a nested dictionary of DataFrames, with file readers dispatched by extension. "
        "The reader registry (`add_reader` / `get_reader`) lets you plug in custom formats "
        "alongside the built-in CSV, JSON, JSONL, and YAML readers.",
    ),
    (
        "MonoSource — `data_conduit.datasources.monosource`",
        "data_conduit.datasources.monosource",
        "Convenience wrappers around `collect_dfs` that preconfigure readers, level selectors, "
        "and named DataArray extraction for common experimental data types. "
        "`MonoSource` is the base class; `FileTypeData` handles flat file formats. "
        "Presets like `ExperimentEvents`, `VideoData`, and `RotationData` provide one-line "
        "loading for standard Bonsai workflow outputs.",
    ),
    (
        "HARP Presets — `data_conduit.datasources.presets.harp`",
        "data_conduit.datasources.presets.harp",
        "HARP-specific device and multi-device presets (requires `harp-python`). "
        "`Device` handles HARP .bin files; `SoundCard`, `CameraStart`, and `Camera0Frames` "
        "are one-line presets for common HARP boards. `MultiDevice` and `Nosepoke` combine "
        "data from multiple HARP boards into unified DataArrays with virtual coordinate lookup.",
    ),
    (
        "MultiSource — `data_conduit.datasources.multisource`",
        "data_conduit.datasources.multisource",
        "Orchestration layer for combining data from multiple devices or files into unified "
        "xarray structures using virtual coordinate maps. `MultiSource` is the base class; "
        "see `data_conduit.datasources.presets.harp` for HARP-specific multi-device presets "
        "(`MultiDevice`, `Nosepoke`).",
    ),
    (
        "Virtual Arrays — `data_conduit.virtualarrays`",
        "data_conduit.virtualarrays",
        "Construct and query n-dimensional xarray DataArrays where *virtual coordinates* "
        "(e.g. device, register, channel) map onto a single *global coordinate* axis. "
        "`ulookup` provides flexible selection by any combination of virtual coordinates, "
        "and the `.ulookup()` xarray accessor makes this available directly on DataArrays.",
    ),
    (
        "Segmentation — `data_conduit.segment`",
        "data_conduit.segment",
        "`get_segment` filters a DataFrame to rows where a lookup column falls within a given "
        "range, returning the corresponding values from a second column as a NumPy array. "
        "Typically used to extract global-clock indices covering a trial window, which are "
        "then passed to `.sel()` or `.loc[]` to slice a DataArray aligned to the same clock.",
    ),
]

SYNC = [
    (
        "TTL Sync — `data_conduit.sync`",
        "data_conduit.sync",
        "End-to-end TTL synchronisation pipeline: extract pulse segments from raw waveforms, "
        "align pulse tables across clocks, fit a linear timebase model, and apply the conversion. "
        "`TTLSyncModel` encapsulates the fitted slope/intercept/R²; "
        "`get_npx_to_bonsai_time_conversion` is a semantic shortcut for the common "
        "Neuropixels ↔ Bonsai alignment. Includes visualisation helpers for inspecting "
        "pulse alignment and conversion error.",
    ),
    (
        "Global Times — `data_conduit.globaltimes`",
        "data_conduit.globaltimes",
        "Create a canonical, evenly-spaced global time vector and map heterogeneous stream "
        "timestamps onto it. `index_map_util` supports nearest, before, after, and exact "
        "matching strategies.",
    ),
]

INTERNAL = [
    (
        "Timestamps — `data_conduit.timestamps`",
        "data_conduit.timestamps",
        "Collect timestamp vectors from individual DataFrames, flat dictionaries, or "
        "arbitrarily nested data structures. Supports flexible lookup by column name, "
        "index, or auto-detection.",
    ),
    (
        "HarpTools — `data_conduit.harptools`",
        "data_conduit.harptools",
        "Extends the `harp-python` library with device reader construction from YAML schemas, "
        "register address lookup, and a `read_harp_bin` reader that plugs into the IO registry.",
    ),
    (
        "Utils — `data_conduit.utils`",
        "data_conduit.utils",
        "Helper callables for level selectors (`starts_with`, `ends_with`, `contains`) and "
        "internal directory-traversal / nested-dict utilities.",
    ),
    (
        "Validators — `data_conduit.validators`",
        "data_conduit.validators",
        "Shared validation logic used by `MonoSource` and `MultiSource` to resolve whether "
        "to build a new `dfs_dict` from a directory or reuse a pre-built one.",
    ),
]


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    out = Path(__file__).parent.parent / "docs" / "API_Reference.md"
    out.parent.mkdir(exist_ok=True)

    content = "\n".join([
        "# `data-conduit` — API Reference",
        "",
        "**Auto-generated from source docstrings** — reflects the current state of the library.",
        "",
        "| Section | What it covers |",
        "|---------|----------------|",
        "| [1. Core Workflow Modules](#1-core-workflow-modules) | IO, MonoSource, HARP Presets, MultiSource, Virtual Arrays, Segmentation |",
        "| [2. Synchronisation & Alignment](#2-synchronisation--alignment) | TTL Sync, Global Times |",
        "| [3. Internals & Utilities](#3-internals--utilities) | Timestamps, HarpTools, Utils, Validators |",
        "",
        "---",
        _render_section(CORE, "1. Core Workflow Modules"),
        "",
        "---",
        _render_section(SYNC, "2. Synchronisation & Alignment"),
        "",
        "---",
        _render_section(INTERNAL, "3. Internals & Utilities"),
    ])

    out.write_text(content, encoding="utf-8")
    print(f"API reference written → {out}")


if __name__ == "__main__":
    main()
