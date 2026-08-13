# Data availability and governance

This repository combines source code, configuration, synthetic experiment
artifacts, real RGB-D recordings, and third-party simulator/model assets. They
do not all have the same release status.

## Public-repository defaults

| Material | Default public treatment | Reason |
|---|---|---|
| Original source code, tests, and software configuration | Apache-2.0 where listed in `LICENSE_SCOPE.md`; suitable for version control after secret/path review | Needed to inspect and reproduce methods |
| Public documentation and non-sensitive aggregate summaries | Suitable for version control, but no separate open-content licence is granted unless stated | Preserve claim boundaries without silently relicensing research content |
| Small protocol locks, manifests, hashes, and aggregate result tables | Suitable for version control | Preserve provenance and negative results |
| Large generated outputs and intermediate caches | Store outside ordinary Git unless a curated small release is prepared | Size, duplication, and environment specificity |
| Real-human SVO/RGB/depth data | Restricted; do not publish by default | Identifiable imagery and consent/privacy obligations |
| Captured environment meshes | Restricted review before release | May reveal a physical space and may have unclear capture pairing |
| ZED/MediaPipe model caches, Isaac assets, and other third-party binaries | Do not redistribute unless their licenses explicitly permit it | Third-party rights and large size |
| Local IDE files, temporary files, logs, and machine-specific paths | Do not publish | No scientific value and possible information leakage |

Repository presence on a workstation does not itself authorize public release.
Before publishing a commit, inspect the staged file list and large-file list.

## Real SVO and OBJ assets

The original local case-study assets are:

| Asset | SHA-256 | Scientific role |
|---|---|---|
| Original real-ZED SVO (restricted local asset) | `5c531c560c71094a44dacfffd807d4f352dc09084d8e94eba79c74631ab4e0e8` | Real ZED input recording; no embedded human GT |
| `mesh.obj` | `518bc40932965c2ef44ee69665b0e82e2de94c5abcc9bc354a053111b782c4d8` | Unregistered static environment mesh, visualization-only |

The SVO contains imagery of a real person and should not be placed in a public
Git repository without a separate, documented release decision covering
consent, privacy, and applicable institutional requirements. Derived images,
depth maps, landmarks, or crops can remain identifiable and require the same
review.

The OBJ does not declare units, camera pose, timestamp, or object labels. Its
held-out registration to the SVO failed four of five gates. It must not be
released or described as human ground truth, and it cannot support clearance
or separation values. Release of the mesh also requires a privacy review of
the captured physical environment.

Hashes allow an authorized recipient to verify a separately transferred
asset. They do not make the underlying data public and do not establish that
two assets share a capture session or coordinate frame.

## Synthetic data and simulator assets

Selected synthetic tables, protocol locks, camera parameters, and derived
metrics may be suitable for a curated data release. Before release:

1. separate original third-party character, scene, texture, model, and USD
   assets from project-generated measurements;
2. check the redistribution terms for every included asset;
3. retain mapping, coordinate-system, unit, and simulator-render provenance;
4. include failed and excluded run status rather than silently deleting it;
5. document whether each table contains measured or inferred joints; and
6. include hashes for the final release bundle.

Isaac USD Skeleton is ground truth only for the synchronized simulation
experiment under its mapping and semantic contract. `BODY_38` output remains
a reference estimate even when generated from synthetic inputs.

## Aggregate and derived data

Aggregate, de-identified result tables are preferred for a first public
release. De-identification should be evaluated rather than assumed: precise
body proportions, trajectories, timestamps, backgrounds, filenames, camera
serial numbers, and metadata may still be identifying or operationally
sensitive.

When raw data cannot be released, a reproducibility bundle can still include:

- frozen protocol and schema;
- input file hashes and non-sensitive metadata;
- code and configuration;
- synthetic or mocked test fixtures;
- aggregate gate tables, including negative outcomes; and
- instructions describing which steps require restricted inputs.

## Access and future releases

No public data DOI or archival dataset identifier is currently declared here.
A future release should receive a versioned manifest and explicit licensing,
consent, and redistribution review. Until then, do not infer access rights from
filenames, hashes, documentation, or local repository placement.

Any data-access request should be evaluated by the repository maintainers
against participant consent, institutional policy, third-party terms, and the
minimum data needed for the proposed replication. This document does not
promise that restricted data can be shared.
