"""Static execution preflight for the Q_C notebooks."""

import json
from pathlib import Path

QC_DIRECTORY = Path(__file__).resolve().parents[2] / "qc"


def _notebooks() -> list[Path]:
    """Return every shipped Q_C notebook in stable order."""
    return sorted(QC_DIRECTORY.glob("*.ipynb"))


def test_every_qc_notebook_code_cell_compiles() -> None:
    """Reject notebook code cells that cannot be executed as Python."""
    notebooks = _notebooks()
    assert notebooks, "no Q_C notebooks were found"

    for notebook in notebooks:
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        for index, cell in enumerate(payload.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(source, f"{notebook.name}:cell-{index}", "exec")


def test_qc_notebooks_do_not_bypass_actual() -> None:
    """Keep notebooks off legacy modules and the retired data mount."""
    forbidden = (
        "data_conduit.datastructure",
        "data_conduit.datasources",
        "data_conduit.sessiongroups",
        "data_conduit.utils",
        "/media/sepi/Elements1/PathIntegrationProtocol",
    )

    for notebook in _notebooks():
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        code = "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []) if cell.get("cell_type") == "code")
        assert not [value for value in forbidden if value in code], notebook.name
