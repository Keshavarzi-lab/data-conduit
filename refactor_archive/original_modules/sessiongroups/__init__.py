'''
Reference session groups: select, load, align, and combine sessions.
--------------------------------------------------------------------

Description:
    1. select_sessions      :  choose session folders from a directory tree at
                              a configurable depth, with All / Include / Exclude
                              rules; label each (default: folder name, or via
                              DateTimeExtractor / LabelExtractor / NameExtractor).
    2. default_harp_catalog / DataStructureCatalog
                               configure WHICH data structures to extract per
                              session (toggle / add / remove specs).
    3. build_sessions       :  load each selected session via the alignment
                              pipeline (load_session, normalise_to_zero,
                              attach_cross_clock, build_global_clock,
                              build_index_tables), returned as a SessionGroup.
    4. combine_sessions     :  stack one structure type (or all) across sessions
                              into a single multi-session structure, tagged by
                              session.

    The bundle primitives, Session, and SessionGroup underpin all of this.
    Alignment is a pipeline of small functions, not a class; callers pick
    whichever steps they need.

Contents:
--------------------------------
- Bundle, Session, SessionGroup     (core data types)
- combine_bundles, map_bundle, select_bundle, select_bundle_where, split_bundle
                                    (bundle primitives)
- select_sessions, LabelExtractor, NameExtractor, DateTimeExtractor
                                    (selection + labelling)
- DataStructureSpec, DataStructureCatalog, default_harp_catalog
                                    (what-to-extract registry)
- load_session, normalise_to_zero, attach_cross_clock,
  build_global_clock, build_index_tables
                                    (alignment pipeline)
- build_sessions, combine_sessions, add_label_column, drop_columns
                                    (orchestration + tailoring)
- source_loader                     (adapter)
'''

from data_conduit.sessiongroups.alignment import (
    attach_cross_clock,
    build_global_clock,
    build_index_tables,
    load_session,
    normalise_to_zero,
)
from data_conduit.sessiongroups.builders import source_loader
from data_conduit.sessiongroups.bundle import (
    Bundle,
    combine_bundles,
    map_bundle,
    select_bundle,
    select_bundle_where,
    split_bundle,
)
from data_conduit.sessiongroups.data_structures import (
    DataStructureCatalog,
    DataStructureSpec,
    default_harp_catalog,
)
from data_conduit.sessiongroups.database import (
    add_label_column,
    build_sessions,
    combine_sessions,
    drop_columns,
)
from data_conduit.sessiongroups.extractors import (
    DateTimeExtractor,
    LabelExtractor,
    NameExtractor,
)
from data_conduit.sessiongroups.selection import select_sessions
from data_conduit.sessiongroups.sessiongroups_core import Session, SessionGroup

__all__ = [
    # core data types
    'Bundle',
    'Session',
    'SessionGroup',
    # bundle primitives
    'combine_bundles',
    'map_bundle',
    'select_bundle',
    'select_bundle_where',
    'split_bundle',
    # selection + labelling
    'select_sessions',
    'LabelExtractor',
    'NameExtractor',
    'DateTimeExtractor',
    # what-to-extract registry
    'DataStructureSpec',
    'DataStructureCatalog',
    'default_harp_catalog',
    # alignment pipeline
    'load_session',
    'normalise_to_zero',
    'attach_cross_clock',
    'build_global_clock',
    'build_index_tables',
    # orchestration + tailoring
    'build_sessions',
    'combine_sessions',
    'add_label_column',
    'drop_columns',
    # adapter
    'source_loader',
]
