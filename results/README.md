# Curated results

This directory is reserved for small, publication-cleared result summaries. Raw
experiment products remain under ignored `output/`; real-person captures, RGB-D
exports, SVO/SVO2 files, and `mesh.obj` must never be copied here.

A curated result contribution should include:

- the producing commit and exact command;
- protocol-lock, input-manifest, and source-artifact hashes;
- method/configuration, units, coordinate frame, joint set, denominator, exclusions,
  and measured/inferred provenance;
- all registered gates, including failed, negative, mixed, or aborted outcomes;
- a machine-readable table (`.csv` or `.json`) plus a concise Markdown explanation;
- confirmation that every included datum and figure is cleared for publication.

Do not commit a derived table solely because it is small. If it cannot be traced to
an immutable input and protocol, keep it local. Curated summaries must not imply
that the portable GitHub CI reproduces ZED hardware, Isaac Sim, private-data, or
full historical experiment workflows.
