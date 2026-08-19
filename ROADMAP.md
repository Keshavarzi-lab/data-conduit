# Roadmap

## Before the next public release

- Run the migrated QC notebooks against the laboratory datasets that are not
  available in the repository.
- Decide whether QC catalog readers for optional modalities should explicitly
  use `optional=True`, and document the intended error policy.
- Choose the next package version; the existing `v0.1.0` tag already refers to
  an earlier revision.
- Review the public API and examples against representative HARP and DeepLabCut
  sessions before treating those integrations as stable.

## Toward a stable API

- Add focused coverage for the documented end-to-end workflows and supported
  Python versions.
- Stabilise extension contracts for readers, catalog configurators, label
  extractors, and datasource integrations.
- Expand examples for common laboratory layouts without embedding private data
  in the package or documentation build.
- Publish migration notes when promoted APIs are renamed or removed.
