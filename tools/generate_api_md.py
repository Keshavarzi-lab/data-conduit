#!/usr/bin/env python3
"""Generate the Sphinx API pages from a static public-module manifest.

The generator deliberately validates module paths on disk rather than importing
them.  API documentation can therefore be checked without installing optional
hardware integrations such as ``harp-python``.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
API_ROOT = ROOT / "docs" / "source" / "api"

PAGES: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "core.rst": (
        "Core API",
        (
            ("Datetime", "data_conduit.core.datetime"),
            ("Global times", "data_conduit.core.globaltimes"),
            ("Input and output", "data_conduit.core.io"),
            ("Segmentation", "data_conduit.core.segment"),
            ("Synchronisation", "data_conduit.core.sync"),
            ("Timestamps", "data_conduit.core.timestamps"),
            ("Utilities", "data_conduit.core.utils"),
            ("Virtual arrays", "data_conduit.core.virtualarrays"),
            ("Validators", "data_conduit.validators"),
        ),
    ),
    "datasources.rst": (
        "Datasource API",
        (
            ("MonoSource", "data_conduit.datasources.monosource"),
            ("MultiSource", "data_conduit.datasources.multisource"),
        ),
    ),
    "datastructures.rst": (
        "Data-structure API",
        (("Data structures", "data_conduit.datastructures"),),
    ),
    "integrations.rst": (
        "Integration API",
        (
            ("DeepLabCut pose", "data_conduit.integrations.DLC.pose"),
            (
                "HARP datasource presets",
                "data_conduit.integrations.harp.datasource_presets",
            ),
            ("HARP tools", "data_conduit.integrations.harp.harptools"),
        ),
    ),
}


def _module_exists(import_path: str) -> bool:
    """Return whether *import_path* maps to a Python module in ``src``."""
    module_path = SOURCE_ROOT.joinpath(*import_path.split("."))
    return module_path.with_suffix(".py").is_file() or (
        module_path / "__init__.py"
    ).is_file()


def _render_page(title: str, modules: tuple[tuple[str, str], ...]) -> str:
    lines = [title, "=" * len(title), ""]
    for heading, import_path in modules:
        lines.extend(
            [
                heading,
                "-" * len(heading),
                "",
                f".. automodule:: {import_path}",
                "   :members:",
                "   :show-inheritance:",
                "",
            ]
        )
    return "\n".join(lines)


def _expected_pages() -> dict[Path, str]:
    missing = [
        import_path
        for _, modules in PAGES.values()
        for _, import_path in modules
        if not _module_exists(import_path)
    ]
    if missing:
        missing_list = "\n".join(f"  - {module}" for module in missing)
        raise RuntimeError(f"API manifest contains missing modules:\n{missing_list}")

    return {
        API_ROOT / filename: _render_page(title, modules)
        for filename, (title, modules) in PAGES.items()
    }


def _check_pages(expected: dict[Path, str]) -> int:
    stale = False
    for path, content in expected.items():
        actual = path.read_text(encoding="utf-8") if path.exists() else ""
        if actual == content:
            continue
        stale = True
        print(f"API page is missing or stale: {path.relative_to(ROOT)}")
        print(
            "".join(
                difflib.unified_diff(
                    actual.splitlines(keepends=True),
                    content.splitlines(keepends=True),
                    fromfile=str(path.relative_to(ROOT)),
                    tofile=f"generated/{path.name}",
                )
            )
        )
    return int(stale)


def _write_pages(expected: dict[Path, str]) -> None:
    API_ROOT.mkdir(parents=True, exist_ok=True)
    for path, content in expected.items():
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when committed API pages differ from the manifest",
    )
    args = parser.parse_args()

    expected = _expected_pages()
    if args.check:
        return _check_pages(expected)
    _write_pages(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
