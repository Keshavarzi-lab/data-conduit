"""Execute movement notebooks in fresh kernels and save separate run artifacts.

Run with the environment containing movement, nbclient, nbformat and ipykernel:
    python tools/validate_movement_notebooks.py --output /path/to/results
Recording selectors come from each notebook's settings cell. Source notebooks
are never overwritten; execution stops at the first failing cell per notebook.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from jupyter_client import KernelManager
import nbformat
from nbclient import NotebookClient


REPO = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = REPO / "src/movement_figures"
AUDIT = '''import json as _validation_json
import data_conduit as _validation_dc
import movement as _validation_movement
import movement_figures.data_template.loading as _validation_loading
assert len(streams['trials']) > 0, "No real trials were loaded."
assert streams['dlc:position'].sizes['Time'] > 0, "No aligned pose frames were loaded."
_validation_summary = {
    "trial_rows": len(streams['trials']),
    "pose_frames": streams['dlc:position'].sizes['Time'],
    "sessions": streams['trials']['session'].drop_duplicates().tolist(),
    "movement_version": _validation_movement.__version__,
    "loading_module": _validation_loading.__file__,
    "data_conduit_module": _validation_dc.__file__,
}
print("NOTEBOOK_VALIDATION=" + _validation_json.dumps(_validation_summary))
'''


def execute(path: Path, output: Path, timeout: int) -> dict:
    """Run the complete notebook, then check that it loaded actual data."""
    relative = path.relative_to(NOTEBOOK_ROOT)
    destination = output / "executed" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    original = path.read_bytes()
    notebook = nbformat.reads(original.decode(), as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    notebook.cells.append(nbformat.v4.new_code_cell(AUDIT, metadata={"tags": ["validation-audit"]}))
    manager = KernelManager(kernel_name="python3")
    # Use exactly the interpreter running this command, regardless of saved metadata.
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(
        notebook, km=manager, timeout=timeout,
        resources={"metadata": {"path": str(path.parent)}},
        record_timing=True,
    )
    start = time.monotonic()
    result = {"notebook": str(relative), "source_sha256": hashlib.sha256(original).hexdigest()}
    print(f"START {relative}", flush=True)
    try:
        client.execute()
        result["status"] = "passed"
    except Exception as error:
        result["status"] = "failed"
        result["exception"] = type(error).__name__
        (destination.with_suffix(".error.txt")).write_text(str(error))
    finally:
        # A supplied KernelManager is owned by this function, not NotebookClient.
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
        nbformat.write(notebook, destination)
    result["seconds"] = round(time.monotonic() - start, 2)
    result["executed_code_cells"] = sum(c.cell_type == "code" and c.execution_count is not None for c in notebook.cells[:-1])
    result["total_code_cells"] = sum(c.cell_type == "code" for c in notebook.cells[:-1])
    result["figure_outputs"] = 0
    for index, cell in enumerate(notebook.cells):
        for item in cell.get("outputs", []):
            if item.output_type == "error":
                result["error_cell"] = index
                result["error"] = f"{item.ename}: {item.evalue}"
            if "image/png" in item.get("data", {}):
                result["figure_outputs"] += 1
            if item.output_type == "stream":
                for line in item.text.splitlines():
                    if line.startswith("NOTEBOOK_VALIDATION="):
                        result["data"] = json.loads(line.split("=", 1)[1])
    if result["status"] == "passed":
        assert result["executed_code_cells"] == result["total_code_cells"]
        assert Path(result["data"]["loading_module"]).is_relative_to(REPO / "src")
        if relative.parts[0] != "data_template" and result["figure_outputs"] == 0:
            result["status"] = "failed"
            result["error"] = "The plotting notebook completed without displaying a figure."
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Paths relative to src/movement_figures; default: every notebook.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300, help="Seconds per cell.")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    os.environ.setdefault("IPYTHONDIR", str(output / "ipython"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(output / "runtime"))
    # Inline output must be captured in the executed notebooks.
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    notebooks = [NOTEBOOK_ROOT / name for name in args.notebooks] if args.notebooks else sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = [pool.submit(execute, path, output, args.timeout) for path in notebooks]
        for future in as_completed(pending):
            results.append(future.result())
            (output / "results.json").write_text(json.dumps(sorted(results, key=lambda r: r["notebook"]), indent=2) + "\n")
    raise SystemExit(0 if all(r["status"] == "passed" for r in results) else 1)


if __name__ == "__main__":
    main()
