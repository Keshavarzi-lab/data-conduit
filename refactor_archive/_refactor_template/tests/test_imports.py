"""Phase A gate: every module imports, every __all__ resolves."""
import importlib
import pkgutil

import pytest

import data_conduit._refactor_template as refactor_root


def _all_modules():
    return [
        m.name
        for m in pkgutil.walk_packages(refactor_root.__path__,
                                       refactor_root.__name__ + ".")
    ]


@pytest.mark.parametrize("modname", _all_modules())
def test_module_imports(modname):
    """Importing each module raises nothing (catches stale paths, cycles)."""
    importlib.import_module(modname)


@pytest.mark.parametrize("modname", _all_modules())
def test_all_names_resolve(modname):
    """Every name in a module's __all__ actually exists on it."""
    mod = importlib.import_module(modname)
    for name in getattr(mod, "__all__", []):
        assert hasattr(mod, name), f"{modname}.{name} in __all__ but missing"
