'''
Adapters from data-conduit source objects to bundle members.
------------------------------------------------------------

Description:
    Thin glue layer. A "source" (a HARP Device / Nosepoke, a FileTypeData
    such as ExperimentEvents, or a bare DataArray / DataFrame) needs to
    become BUNDLE MEMBERS (the {name: object} entries the rest of the
    package operates on). ``_object_to_bundle`` does that coercion;
    ``source_loader`` wraps it as a ``loader(session_path) -> bundle``
    convenience.

    ``_object_to_bundle`` is reused by alignment.py (which calls it on each
    DataStructureSpec's loaded object). Session SELECTION is handled
    separately by selection.py (directory walk), not here.

Contents:
--------------------------------
- _object_to_bundle:    Coerce one loaded source object into bundle members.
- source_loader:        Build a session-id -> bundle loader from source classes.
'''





################################################################################
# Imports
################################################################################

from collections.abc import Callable, Mapping

import pandas as pd
import xarray as xr

from data_conduit.sessiongroups.bundle import Bundle

################################################################################




################################################################################
# Source-Object -> Bundle Members
################################################################################



#===============================================================================
# 1| Coerce One Loaded Source Object into Bundle Members
#===============================================================================
def _object_to_bundle(
        obj,
        name: str,
        *,
        prefix_data_arrays: bool,
) -> Bundle:
    '''
    Coerce one loaded source object into bundle members.

    A "source" can be several shapes; this normalises each to
    ``{name: object}`` entries. The checks are ordered most- to
    least-specific so the right branch fires before a more permissive one:

      1. an object with a ``.data_arrays`` mapping (MonoSource / MultiSource,
         e.g. SoundCard / Nosepoke) -> each array becomes a member,
         optionally prefixed with the source name;
      2. an object with a ``.df`` DataFrame (e.g. ExperimentEvents) -> one
         member;
      3. a bare DataArray / Dataset / DataFrame -> one member;
      4. a plain mapping of those -> members, optionally prefixed.

    ----------
    Parameters:
        obj (object):
            The loaded source object to coerce. Accepts any of the four
            shapes listed above; raises TypeError otherwise.
        name (str):
            The source name. Becomes the member key (single-member shapes)
            or the prefix (multi-member shapes).
        prefix_data_arrays (bool):
            If True, prefix each entry from ``.data_arrays`` or from a plain
            mapping with ``f'{name}:'`` so members coming from several
            sources don't collide on shared sub-names (e.g.
            ``'nosepoke:Activations'`` vs ``'soundcard:Activations'``).
    Returns:
        Bundle:
            A dict mapping member name to data object, ready to merge into
            the rest of the bundle.
    '''

    # 1| Source instance exposing named DataArrays (the common HARP case).
    data_arrays = getattr(obj, 'data_arrays', None)
    if isinstance(data_arrays, Mapping):
        # Prefix to avoid key collisions when several sources are merged.
        if prefix_data_arrays:
            return {f'{name}:{key}': value for key, value in data_arrays.items()}
        return dict(data_arrays)

    # 2| Source instance exposing a single DataFrame (e.g. ExperimentEvents).
    df = getattr(obj, 'df', None)
    if isinstance(df, pd.DataFrame):
        return {name: df}

    # 3| Already a bare data object.
    if isinstance(obj, xr.DataArray | xr.Dataset | pd.DataFrame):
        return {name: obj}

    # 4| A plain mapping of data objects.
    if isinstance(obj, Mapping):
        if prefix_data_arrays:
            return {f'{name}:{key}': value for key, value in obj.items()}
        return dict(obj)

    # No branch matched: caller passed an unsupported shape.
    raise TypeError(
        f"source {name!r} produced {type(obj).__name__}; expected an object with "
        f'.data_arrays/.df, a DataArray/Dataset/DataFrame, or a mapping of those.'
    )

#===============================================================================



#===============================================================================
# 2| Build a Session-ID-Driven Loader From Source Classes
#===============================================================================
def source_loader(
        *sources: tuple[str, Callable[..., object]],
        path_resolver: Callable[..., object],
        prefix_data_arrays: bool = True,
) -> Callable[..., Bundle]:
    '''
    Build a ``loader(session_id, **path_keys) -> bundle`` from source classes.

    Convenience for callers who want one function that, given a session id
    (and optional path keys), resolves the on-disk path and loads several
    sources into one bundle. The alignment-pipeline workflow does not need
    this (it loads via a DataStructureCatalog), but it remains a handy
    standalone adapter for ad-hoc analyses.

    ----------
    Parameters:
        *sources (tuple[str, callable]):
            ``(name, factory)`` pairs, where ``factory(path) -> source``
            builds a source object from a resolved session path, e.g.
            ``('events', lambda p: ExperimentEvents(experiment_directory_path=p))``.
        path_resolver (callable):
            ``path_resolver(session_id, **path_keys) -> path``. Encapsulates
            the on-disk layout (kept caller-supplied so the package makes
            no assumptions about folder shape).
        prefix_data_arrays (bool):
            If True (default), prefix ``.data_arrays`` member names with the
            source name to avoid collisions across sources.
    Returns:
        callable:
            A ``loader(session_id, **path_keys) -> Bundle``.
    '''

    def loader(session_id: str, **path_keys) -> Bundle:
        '''
        Resolve the path for ``session_id`` and merge every source into one bundle.

        ----------
        Parameters:
            session_id (str):
                The session identifier passed to ``path_resolver``.
            **path_keys:
                Extra keyword arguments forwarded to ``path_resolver`` (for
                resolvers that need more than just the session id).
        Returns:
            Bundle:
                The merged bundle of members from every source.
        '''

        # 1| Turn the id + path keys into a concrete session path.
        path = path_resolver(session_id, **path_keys)

        # 2| Build each source from that path and merge its members into one
        #    bundle. Later sources overwrite earlier ones on key collision
        #    (the prefix above is designed to prevent that).
        bundle: Bundle = {}
        for name, factory in sources:
            bundle.update(
                _object_to_bundle(factory(path), name, prefix_data_arrays=prefix_data_arrays)
            )
        return bundle

    return loader

#===============================================================================



################################################################################
