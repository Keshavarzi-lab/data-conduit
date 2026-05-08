'''Notebook setup helpers for the design-space analysis notebooks.'''

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / 'src'


def add_repo_paths() -> None:
    '''Make the local workflow and src-layout package importable.'''
    for path in (REPO_ROOT, SRC_ROOT):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


add_repo_paths()


__all__ = [
    'REPO_ROOT',
    'SRC_ROOT',
    'add_repo_paths',
]
