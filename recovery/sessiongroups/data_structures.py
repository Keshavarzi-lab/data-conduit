'''
Configurable registry of WHICH data structures to extract from a session.
-------------------------------------------------------------------------

Description:
    A single experimental session folder contains many kinds of data:
    nosepoke activations, soundcard logs, experiment events, camera frames,
    video metadata, and so on. When we load a session (see alignment.py)
    we rarely want a fixed, hard-coded set: we want to choose which
    structures to look for, switch some off temporarily, or add a new one,
    without editing the loader.

    This module is that control surface. It is pure configuration: NOTHING
    here reads any data. It only describes what to read; the loader in
    alignment.py does the actual work by walking this description.

    Lifecycle:
        catalog = default_harp_catalog('device.yml', 'soundcard.yml')  # common set
        catalog.disable('video')                                       # tweak
        session = load_session(session_path, catalog)                  # loader reads enabled specs

Contents:
--------------------------------
- DataStructureSpec:        Record describing one data structure to load.
- DataStructureCatalog:     Ordered, editable collection of specs.
- default_harp_catalog:     Pre-filled catalog of the lab's common structures.
'''





################################################################################
# Imports
################################################################################

# ``dataclass`` turns a class into a simple record: list the fields and
# Python writes __init__/__repr__/equality for you. Perfect for a small
# settings record.
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

################################################################################




################################################################################
# DataStructureSpec
################################################################################



#===============================================================================
# 1| DataStructureSpec Dataclass
#===============================================================================
@dataclass
class DataStructureSpec:
    '''
    A description of one data structure to (try to) extract from a session.

    A spec answers four questions about a single data structure: what is it
    called, how do I load it, should I load it right now, and what happens
    if it is missing. The ``@dataclass`` decorator means you create one
    just by naming the fields, e.g.:

        DataStructureSpec(
            name='soundcard',
            reader=lambda p: SoundCard(experiment_directory_path=p),
        )

    ----------
    Parameters:
        name (str):
            Short identifier for this structure, e.g. ``'nosepoke'`` or
            ``'events'``. Also becomes the PREFIX of that structure's
            members once loaded: the loader merges every structure into one
            bundle and prefixes keys with the name to avoid clashes (so a
            Nosepoke's ``Activations`` array appears as
            ``'nosepoke:Activations'``). Names must be unique within a
            catalog.
        reader (callable):
            How to load this structure. A function that takes the session
            directory as a ``Path`` and returns one of:
              * a source object exposing ``.data_arrays`` (a dict of
                arrays) or ``.df`` (a DataFrame), e.g. a ``SoundCard`` or
                ``ExperimentEvents``;
              * a bare xr.DataArray / xr.Dataset / pd.DataFrame;
              * a plain dict mapping names to those.
            In practice it is a tiny lambda wrapping a preset class, e.g.
            ``lambda p: SoundCard(experiment_directory_path=p,
            harp_device_yaml_path=...)``. The loader knows how to turn each
            of these return shapes into bundle members; see
            ``builders._object_to_bundle``.
        enabled (bool):
            Whether this spec is currently switched on. The loader only
            processes enabled specs, so setting this to False is how you
            temporarily leave a structure out without deleting it from the
            catalog. Default True.
        required (bool):
            What to do when this structure cannot be loaded (its files are
            absent, or its reader raises/returns nothing):
              * True  -> hard error; loading the session fails.
              * False (default) -> quietly skipped (with a warning).
            False gives the "load it only if it's present" behaviour, which
            is what you want for optional modalities that some sessions lack.
        sync (dict | None):
            How this structure's CLOCK relates to the session's main clock.
              * None (default) -> the data already lives on the session's
                HARP/Bonsai clock, so no conversion is needed.
              * a dict -> this is a CROSS-CLOCK structure (e.g. Neuropixels
                or DLC) recorded on its own clock. ``attach_cross_clock``
                must first convert its times onto the reference clock (via
                TTL sync) before merging it; the dict carries the
                configuration the loader forwards to the TTL-sync
                utilities. This field is what the loader checks to decide
                "same-clock" vs "needs conversion".
    '''

    name: str
    reader: Callable[[Path], object]
    enabled: bool = True
    required: bool = False
    sync: dict | None = None

#===============================================================================



################################################################################




################################################################################
# DataStructureCatalog
################################################################################



#===============================================================================
# 1| DataStructureCatalog Class
#===============================================================================
class DataStructureCatalog:
    '''
    An ordered, editable collection of DataStructureSpec objects.

    Think of it as the shopping list of data structures the loader should
    look for, IN ORDER. Unlike most objects in this package (which are
    immutable and return copies), a catalog is deliberately MUTABLE: it is
    a configuration object you build and then tweak in place with ``add`` /
    ``remove`` / ``enable`` / ``disable`` before handing it to the loader.

    Internally it is a name -> spec mapping, so two things follow: spec
    names must be unique, and iteration order is the order specs were
    added (Python's plain dict preserves insertion order, which is what
    the loader uses).

    ----------
    Parameters:
        specs (iterable of DataStructureSpec):
            Initial specs, kept in the given order. Default: empty.
    '''

    #---------------------------------------------------------------------------
    # 1.1| Construction
    #---------------------------------------------------------------------------
    def __init__(
            self,
            specs=(),
    ) -> None:
        '''
        Store the initial specs in an order-preserving name -> spec mapping.

        ----------
        Parameters:
            specs (iterable of DataStructureSpec):
                Specs to seed the catalog with.
        Returns:
            None.
        '''
        # Start empty, then add() each one. Reusing add()'s duplicate-name
        # check avoids blindly trusting the input.
        self._specs: dict[str, DataStructureSpec] = {}
        for spec in specs:
            self.add(spec)


    #---------------------------------------------------------------------------
    # 1.2| Editing (add / remove / enable / disable, all chainable)
    #---------------------------------------------------------------------------
    def add(
            self,
            spec: DataStructureSpec,
    ) -> 'DataStructureCatalog':
        '''
        Add a spec to the end of the catalog.

        Returns ``self`` so calls can be chained:
        ``catalog.add(s1).add(s2)``.

        ----------
        Parameters:
            spec (DataStructureSpec):
                The spec to add.
        Returns:
            DataStructureCatalog:
                The catalog itself (for chaining).
        '''
        # Names key the mapping; a duplicate would silently overwrite, so
        # guard against that and surface the clash explicitly.
        if spec.name in self._specs:
            raise ValueError(f'a data structure named {spec.name!r} is already in the catalog.')
        self._specs[spec.name] = spec
        return self

    def remove(
            self,
            name: str,
    ) -> 'DataStructureCatalog':
        '''
        Remove a spec by name entirely (returns self).

        Removing a name that isn't present is a harmless no-op, not an error.

        ----------
        Parameters:
            name (str):
                The spec name to remove.
        Returns:
            DataStructureCatalog:
                The catalog itself (for chaining).
        '''
        # pop with a default makes "remove unknown name" a no-op rather than
        # raising KeyError.
        self._specs.pop(name, None)
        return self

    def enable(
            self,
            name: str,
    ) -> 'DataStructureCatalog':
        '''
        Switch a spec on by name (returns self).

        ----------
        Parameters:
            name (str):
                The spec name to enable. Raises KeyError if unknown.
        Returns:
            DataStructureCatalog:
                The catalog itself (for chaining).
        '''
        self._set_enabled(name, True)
        return self

    def disable(
            self,
            name: str,
    ) -> 'DataStructureCatalog':
        '''
        Switch a spec off by name without removing it (returns self).

        ----------
        Parameters:
            name (str):
                The spec name to disable. Raises KeyError if unknown.
        Returns:
            DataStructureCatalog:
                The catalog itself (for chaining).
        '''
        self._set_enabled(name, False)
        return self

    def _set_enabled(
            self,
            name: str,
            value: bool,
    ) -> None:
        '''
        Flip one spec's ``enabled`` flag, erroring if the name is unknown.

        Shared by ``enable`` / ``disable``. We toggle the flag on the
        stored spec in place (the catalog is mutable by design); the
        loader later reads this flag to decide whether to process the spec.

        ----------
        Parameters:
            name (str):
                The spec name whose enabled flag we are flipping.
            value (bool):
                New value for the enabled flag.
        Returns:
            None.
        '''
        if name not in self._specs:
            raise KeyError(f'no data structure named {name!r} in the catalog.')
        self._specs[name].enabled = value


    #---------------------------------------------------------------------------
    # 1.3| Reading (enabled_specs / names)
    #---------------------------------------------------------------------------
    def enabled_specs(self) -> list[DataStructureSpec]:
        '''
        Return the currently-enabled specs, in catalog order.

        This is the method the loader calls. Filtering out disabled specs
        here means the loader never has to think about the ``enabled`` flag
        itself.

        Returns:
            list[DataStructureSpec]:
                Enabled specs in insertion order.
        '''
        return [spec for spec in self._specs.values() if spec.enabled]

    @property
    def names(self) -> list[str]:
        '''
        Return all spec names (enabled or not), in order.

        Returns:
            list[str]:
                Spec names in insertion order.
        '''
        return list(self._specs)


    #---------------------------------------------------------------------------
    # 1.4| Convenience Dunders (__iter__ / __len__ / __contains__ / __getitem__ / __repr__)
    #---------------------------------------------------------------------------
    def __iter__(self):
        '''
        Iterate over all specs (enabled or not), in insertion order.

        Returns:
            Iterator over DataStructureSpec objects.
        '''
        return iter(self._specs.values())

    def __len__(self) -> int:
        '''
        Return the number of specs in the catalog.

        Returns:
            int:
                Number of specs.
        '''
        return len(self._specs)

    def __contains__(
            self,
            name: str,
    ) -> bool:
        '''
        Whether a spec with this name exists.

        Enables ``'nosepoke' in catalog``.

        ----------
        Parameters:
            name (str):
                The spec name to look up.
        Returns:
            bool:
                True if the catalog contains a spec with that name.
        '''
        return name in self._specs

    def __getitem__(
            self,
            name: str,
    ) -> DataStructureSpec:
        '''
        Fetch a spec by name.

        Enables ``catalog['nosepoke']``.

        ----------
        Parameters:
            name (str):
                The spec name to look up.
        Returns:
            DataStructureSpec:
                The matching spec. Raises KeyError if absent.
        '''
        return self._specs[name]

    def __repr__(self) -> str:
        '''
        Show each spec's name, marking disabled ones with ``(off)``.

        Returns:
            str:
                A short repr of the form ``<DataStructureCatalog [a, b(off), c]>``.
        '''
        items = ', '.join(
            f'{spec.name}{"" if spec.enabled else "(off)"}' for spec in self._specs.values()
        )
        return f'<DataStructureCatalog [{items}]>'

#===============================================================================



################################################################################




################################################################################
# Default Catalog for the Lab's Common HARP/Bonsai Structures
################################################################################



#===============================================================================
# 1| default_harp_catalog (Builds a Pre-Filled Catalog)
#===============================================================================
def default_harp_catalog(
        device_yaml: str | Path = './device.yml',
        soundcard_yaml: str | Path = './soundcard.yml',
        *,
        include_video: bool = True,
        include_session_settings: bool = True,
) -> DataStructureCatalog:
    '''
    Build a catalog of the lab's common HARP/Bonsai data structures.

    Convenience starting point you call explicitly (it is never a hidden
    default); once you have it you can ``disable`` / ``remove`` / ``add``
    to suit a particular analysis. It assembles, in order:

      * events            :  ExperimentEvents CSV (REQUIRED: it provides
                            the master Bonsai clock everything else is
                            aligned to).
      * nosepoke          :  the 6-board / 18-port Nosepoke MultiDevice.
      * soundcard         :  SoundCard registers.
      * camera            :  Camera0Frames frame counts.
      * video             :  VideoData CSV metadata (optional).
      * session_settings  :  SessionSettings (optional).

    All non-events specs are ``required=False``, so any structure missing
    from a given session is skipped rather than raising.

    ----------
    Parameters:
        device_yaml (str | Path):
            HARP device YAML for the Behavior boards / cameras. Default
            ``'./device.yml'``.
        soundcard_yaml (str | Path):
            HARP device YAML for the SoundCard. Default ``'./soundcard.yml'``.
        include_video (bool):
            If True (default), include the optional VideoData CSV spec.
        include_session_settings (bool):
            If True (default), include the optional SessionSettings spec.
    Returns:
        DataStructureCatalog:
            The assembled, editable catalog.
    '''

    # 1| Import the source classes LAZILY, inside the function, for two reasons:
    #    a) the HARP presets need the optional harp-python package, and we don't
    #       want importing this module to fail when it isn't installed;
    #    b) it keeps the package's import graph light for callers who only use
    #       the spec/catalog types above and never call this helper.
    from data_conduit.datasources.monosource import (
        ExperimentEvents,
        SessionSettings,
        VideoData,
    )
    from data_conduit.datasources.presets.harp import (
        Camera0Frames,
        Nosepoke,
        SoundCard,
    )
    from data_conduit.utils.utils_core import _concat_split_dataframes

    # 1b| FileType .df returns a DataFrame for a single-file session, or a nested
    #     dict mapping file-stem -> DataFrame when a session has multiple files
    #     (e.g. an ExperimentEvents CSV split by a mid-session restart). We want ONE
    #     DataFrame per session per spec so combine_sessions' name-based intersection
    #     lands the structure. ``_concat_split_dataframes`` collapses any such dict
    #     into one frame, joined by FILENAME FIRST then by Time WITHIN each file, with
    #     NO global time sort (so a clock rollback between files stays visible as a
    #     non-monotonic index, and is warned about). See its definition in utils_core
    #     for the full rationale.

    # 2| Each spec wraps a preset in a tiny lambda of the form
    #    ``lambda p: Preset(...)``. The lambda takes the session path ``p``
    #    (supplied later, per session, by the loader) and returns the loaded
    #    source. We capture the YAML paths here so the readers stay strictly
    #    one-argument (path -> source), the contract the loader relies on.
    #
    #    Note: 'events' is required=True because it provides the reference
    #    clock the whole session is normalised / aligned to; a session with no
    #    events cannot be aligned, so we want a clear error rather than a
    #    silent skip.
    #
    #    Return-shape note:
    #      * FileType structures (events / video / session_settings) return
    #        their ``.df`` DataFrame directly. It is the natural,
    #        Time-indexed table interface and keeps that member a clean
    #        DataFrame in the bundle.
    #      * HARP structures (nosepoke / soundcard / camera) return the whole
    #        source OBJECT, so the loader reads its ``.data_arrays``
    #        (DataArrays for MultiDevice/Nosepoke, Datasets for
    #        MonoDevice/SoundCard).
    specs = [
        DataStructureSpec(
            name='events',
            reader=lambda p: _concat_split_dataframes(ExperimentEvents(experiment_directory_path=p).df),
            required=True,
        ),
        DataStructureSpec(
            name='nosepoke',
            reader=lambda p: Nosepoke(experiment_directory_path=p, harp_device_yaml_path=device_yaml),
        ),
        DataStructureSpec(
            name='soundcard',
            reader=lambda p: SoundCard(experiment_directory_path=p, harp_device_yaml_path=soundcard_yaml),
        ),
        DataStructureSpec(
            name='camera',
            reader=lambda p: Camera0Frames(experiment_directory_path=p, harp_device_yaml_path=device_yaml),
        ),
    ]

    # 3| Optional CSV structures, appended only if the caller asked for them.
    #    Both are left required=False so sessions lacking them load fine.
    if include_video:
        specs.append(
            DataStructureSpec(
                name='video',
                reader=lambda p: _concat_split_dataframes(VideoData(experiment_directory_path=p).df),
            )
        )
    if include_session_settings:
        # SessionSettings returns {'metadata': df, 'trials': df}, two semantically
        # distinct sub-tables with different schemas. Leave them split so they
        # land as 'session_settings:metadata' and 'session_settings:trials'.
        specs.append(
            DataStructureSpec(
                name='session_settings',
                reader=lambda p: SessionSettings(experiment_directory_path=p).df,
            )
        )

    return DataStructureCatalog(specs)

#===============================================================================



################################################################################
