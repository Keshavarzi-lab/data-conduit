"""Import contract for every staged package module."""

import importlib
import pkgutil
import subprocess
import sys

import data_conduit.actual as actual


def test_all_actual_modules_import() -> None:
    """Every discoverable module in the staged tree should import cleanly."""
    failures: list[str] = []

    for module in pkgutil.walk_packages(actual.__path__, actual.__name__ + "."):
        try:
            importlib.import_module(module.name)
        except Exception as error:  # pragma: no cover - assertion reports module/error
            failures.append(f"{module.name}: {type(error).__name__}: {error}")

    assert not failures, "\n".join(failures)


def test_lookup_accessor_coexists_with_legacy_in_either_import_order() -> None:
    """The global xarray accessor must register once without namespace warnings."""
    actual_module = "data_conduit.actual.core.virtualarrays.custom_lookup_template"
    legacy_module = "data_conduit.virtualarrays.custom_lookup_template"

    for first, second in ((actual_module, legacy_module), (legacy_module, actual_module)):
        script = (
            "import importlib, warnings, xarray as xr; "
            "warnings.simplefilter('error'); "
            f"first = importlib.import_module({first!r}); "
            f"second = importlib.import_module({second!r}); "
            "installed = xr.DataArray.__dict__['ulookup']._accessor; "
            "assert first.LookupAccessorConstructor is installed; "
            "assert second.LookupAccessorConstructor is installed"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
