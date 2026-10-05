# Public code update scope

Publication synchronization: 2026-10-05. This is a curated research release,
not a mirror of the private workstation or a new experimental execution.

## Added components

The update includes dependency-closed, publication-cleared Python components
for the following work:

- temporal bias/variance and rendered-occlusion analysis;
- BODY_38 comparison and stereo-depth quality analysis;
- conditional View C pre-roll, repeatability and performance aggregation;
- NCC frame-attribution research utilities;
- CTS5 scalar limb-length fusion;
- FS-CTS5 camera geometry, major-limb motion, ankle compensation, retargeting
  and known-occluder handling;
- industrial workcell layout, ground-contact calculations and robot-occluder
  trajectory construction;
- common-bank rules, randomized schedules and selector versions;
- successor probe scheduling and selection logic; and
- reference agreement and depth-sampling replay.

These are method and analysis components. A pure geometry utility, selector,
or robot-occluder trajectory is not evidence of successful adaptive placement,
cross-character generalization, or robot-human shared manipulation.

## Test commands

```powershell
python -m pip install -r requirements/ci.txt
python -m compileall -q tools tests
python tests/portable_suite.py
python tests/research_components_suite.py
```

The original portable allowlist remains separate from the supplemental
research-components allowlist. Both use synthetic fixtures and dependency-light
checks, not restricted captures. Only these explicit allowlists are the hosted
CI contract; do not substitute unrestricted test discovery.

Before publication, the exact staged tree was exported into an isolated
directory without private `output/` artifacts. Python source compilation passed;
the original 23-module suite passed 213 tests, and the supplemental 17-module
suite passed 400 tests (613 in total). Relative documentation links and the
published FS-CTS5 gate fields were also checked against their sources. This
validation does not change any registered scientific pass/fail decision.

## Files not released

Workstation-specific launchers and source/configuration files with local paths,
device identifiers, real-participant profiles, or missing restricted dependencies
remain local. They were not rewritten to bypass publication review. Historical
protocols and their complete private inputs are not promised as runnable from
this repository alone.

Reports and manuscript staging directories remain local because their figures,
captures and third-party redistribution scope need a separate publication review.
The report's progress and aggregate experimental outcomes are published in the
[progress snapshot](project_progress_cn.md), [matrix](final_experiment_matrix.md)
and [results overview](results_overview.md).

The repository's `.gitattributes` defines public text newline normalization.
An original frozen workstation file hash therefore must not be assumed to match
a Git-normalized checkout. Original artifacts were not edited by this update;
public release identity is established by the Git commit, and private-study
identity remains governed by its original locks/manifests.
