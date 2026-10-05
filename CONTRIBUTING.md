# Contributing

Contributions that improve reproducibility, portability, documentation, tests, or
the BlazePose + ZED RGB-D pipeline are welcome. Keep claims and code changes
consistent with the registered experiment boundaries.

## Research invariants

- BlazePose + ZED RGB-D is the baseline. ZED SDK Body Tracking is a
  reference/comparison, not the baseline or ground truth.
- Preserve measured and inferred joint provenance. K4 completion must not
  overwrite or be reported as measured data.
- Do not rewrite frozen inputs, protocol locks, manifests, content hashes, or
  registered negative/mixed outcomes. Add a new versioned derivation instead.
- Use the registered metric name, joint set, denominator, coordinate convention,
  and unit conversion. Do not generalise single-person or simulation evidence to
  completed multi-person or real-world validation.
- Real-person captures, local meshes, restricted simulator assets, camera serial
  numbers, and workstation-specific paths must not enter a pull request.

## Development setup

Use Python 3.11. For the hosted-CI-equivalent environment:

```powershell
python -m pip install -r requirements/ci.txt
python -m compileall -q tools tests
python tests/portable_suite.py
python tests/research_components_suite.py
```

Install `requirements/core.txt`, `requirements/analysis.txt`, or
`requirements/documents.txt` only when the change needs that profile. ZED SDK and
Isaac Sim dependencies are supplied by their own managed environments and must not
be added as ordinary PyPI dependencies.

## Test boundaries

`tests/portable_suite.py` explicitly loads 23 audited hardware-free modules
(213 tests at the initial public release). This is the required GitHub CI
scope; adding a test file does not silently add it to that allowlist.

`tests/research_components_suite.py` adds a separate explicit allowlist for
publication-cleared numerical and selector components. It is also run in hosted
CI and uses synthetic fixtures, not private captures. Neither suite establishes
scientific replication of a frozen study.

The public repository intentionally omits hardware-only tests, governed input
artifacts, and some frozen-lineage checks from the private research workspace.
Passing public discovery or the portable suite therefore validates the curated
software release, not every private capture workflow or scientific result.

## Pull requests

Keep changes focused. Explain affected research claims, list exact validation
commands, and record tests that were skipped or unavailable. Prefer configurable
repository-relative paths for maintained code while leaving hash-audited historical
scripts intact. New hardware-dependent tests should fail or skip with an explicit
reason and remain outside portable CI until a hardware-free fixture exists.

Unless explicitly marked "Not a Contribution" or governed by a separate written
agreement, software contributions intentionally submitted for inclusion are
accepted under Apache-2.0, consistent with Section 5 of the licence. See
`LICENSE_SCOPE.md` before submitting documentation, data, models, or third-party
assets; the software licence does not automatically cover those materials.
