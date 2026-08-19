# Changelog

All notable changes to `data-conduit` will be documented here.

## Unreleased

### Changed

- Promoted the refactored runtime from the temporary `data_conduit.actual`
  namespace into the public `data_conduit.core`, `data_conduit.datasources`,
  `data_conduit.datastructures`, and `data_conduit.integrations` packages.
- Archived the superseded implementation and refactor template under
  `refactor_archive/` so their Git history remains available during review.
- Updated QC helpers and notebooks to import the promoted package paths without
  executing data-dependent notebook cells.
- Rebuilt the GitHub Pages overview, workflow guide, example gallery, data-flow
  visual, and Sphinx-native API reference.
- Added release metadata and excluded laboratory notebooks, generated figures,
  and packaged test material from distributions while retaining QC Python
  helpers.

### Fixed

- Applied the reviewed refactor fixes recorded in the
  [bug-fix line ledger](https://github.com/Keshavarzi-lab/data-conduit/blob/main/BUG_FIX_LINE_LEDGER.md).
