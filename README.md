# BlazeZED-HRC: BlazePose–ZED RGB-D 3D Human Pose and Kinematic Parameter Estimation for Human–Robot Collaboration

BlazeZED-HRC is a public research-code release for estimating human joint
positions, limb lengths, and kinematic constraints from ZED RGB-D observations
in human-robot-collaboration (HRC) settings.

Formal project title: **Parameter Identification and Human Body
Characterisation for Human-Robot Collaboration**.

The project baseline is **BlazePose + ZED RGB-D**. ZED SDK `BODY_38` appears
only as a reference/comparison method; it is neither the baseline nor ground
truth. In simulation, the synchronized Isaac USD Skeleton is the ground truth.

> 中文简介：本项目研究面向人机协作的 RGB-D 人体参数估计。公开结论同时保留准确度、
> 覆盖率、稳定性、实时性及失败恢复结果，并如实报告阴性和混合实验。

## Research scope

The processing chain is:

```text
ZED RGB-D capture
-> BlazePose 2D landmarks
-> confidence and visibility handling
-> robust depth sampling
-> 2D-to-3D back-projection
-> measured 3D joints
-> optional kinematic constraint or inferred completion
-> limb lengths, trajectories, and performance metrics
```

The current core studies are predominantly **single-person** experiments.
Multi-person front-end work exists, but the main simulation, ablation, and
confirmatory evidence must not be generalized to validated multi-person HRC.

The default constrained output is **K2g**, a guarded eight-bone projection of
valid measured joints. **K4** is an optional robustness output for selected
missing right-arm measurements. K4 has explicit `inferred` provenance and
never overwrites valid measured data.

## Current evidence at a glance

Public progress synchronized on **2026-10-05**, covering the completed studies
and latest local artifacts available through **2026-08-31**. Synchronization is
not a new experiment. See the [Chinese progress snapshot](docs/project_progress_cn.md)
and [final experiment matrix](docs/final_experiment_matrix.md).

- A synchronized Isaac Sim exporter provides RGB, metric depth, camera
  parameters, and USD Skeleton ground truth.
- The mapping contract contains 15 canonical joints and 13 core evaluation
  joints.
- Depth sampling, kinematic constraints, temporal tracking, rendered
  occlusion, best-view selection, streaming, and real-SVO replay have been
  evaluated in separate frozen studies.
- Several useful component results passed their registered gates, including
  measured-first best-view selection and conditional View C streaming.
- Several important overall contracts did **not** pass. These include the
  clean temporal dynamic contract, fresh 960x600 static temporal contract,
  `BODY_38` comparison replication direction, dual-view real-time gate,
  adaptive-camera accuracy ranking, and the registered robustness gate in the
  weighted limb-length studies.
- The FS-CTS5 eight-major-limb supplement completed 20/20 formal bundles and
  24,000 camera frames. Five gate families passed, but G3 failed clean
  right-thigh P95 non-inferiority; the overall registered contract is negative.
- Independent cross-character recoveries passed replay exactness checks but
  found no qualifying camera bank under their frozen selectors. Successor-v4
  completed 12/12 raw sessions and froze replay-only authority; no replay or
  downstream scientific result has been recorded for that cohort.
- Registration of the supplied `mesh.obj` to the real SVO failed held-out
  validation. The mesh is therefore visualization-only; this project reports
  no joint-to-mesh clearance or separation values from that asset.

See [Results overview](docs/results_overview.md) for numbers and claim
boundaries.

## Repository map

```text
configs/   publication-cleared method settings and profiles
docs/      curated architecture, results, and reproducibility guides
models/    model acquisition notes; third-party task bundles stay local
output/    local-only experiment artifacts (ignored and not published)
results/   publication-cleared summaries only
tests/     audited, dependency-light unit and integrity tests
tools/     publication-cleared numerical, validation, and protocol utilities
```

Start with:

- [Documentation index](docs/index.md)
- [Architecture and data contracts](docs/architecture.md)
- [Results overview](docs/results_overview.md)
- [Current progress (Chinese)](docs/project_progress_cn.md)
- [Final experiment matrix](docs/final_experiment_matrix.md)
- [Public code update scope](docs/public_code_update.md)
- [Reproducibility guide](docs/reproducibility.md)
- [Data availability and privacy](DATA_AVAILABILITY.md)
- [Software licence and scope](LICENSE_SCOPE.md)
- [Contribution guide](CONTRIBUTING.md)

## Minimal local checks

The repository contains both dependency-light analysis code and
hardware/application-specific capture paths. For the audited hosted-CI scope,
use Python 3.11 and the minimal dependency profile:

```powershell
python -m pip install -r requirements/ci.txt
python -m compileall -q tools tests
python tests/portable_suite.py
python tests/research_components_suite.py
```

Additional paths may require MediaPipe/BlazePose, OpenCV, the ZED SDK and
`pyzed`, or Isaac Sim. The current real-ZED experiments were executed with ZED
SDK/`pyzed` 5.4.0. Follow the frozen protocol associated with an experiment;
do not infer a runnable command solely from a result table. See the
[dependency profiles](requirements/README.md) before installing an SDK-specific
environment.

## Evaluation terminology

- **Measured coverage** counts only joints backed by a valid depth
  measurement. Inferred K4 output is reported separately.
- The dynamic right-arm primary endpoint contains only the right elbow and
  right wrist and is named
  `right_arm_two_joint_mean_position_error_mm`. It is **not** full-body MPJPE.
- Core full-body MPJPE uses the 13-joint contract. Semantically different
  points such as neck and nose are not silently added.
- BlazePose landmarks lie near visible body surfaces, while Isaac ground truth
  is a skeleton pivot. Raw errors therefore contain a semantic offset.
- Offline replay throughput is not camera-to-output FPS. Live reports must
  distinguish source timestamp FPS, acquisition FPS, output FPS, consumer
  compute latency, queue latency, and host latency.

## Software licence

Original software authored by **Zhichao Huang** is licensed under the
[Apache License 2.0](LICENSE). The grant is deliberately limited to the
software and software-support files identified in
[LICENSE_SCOPE.md](LICENSE_SCOPE.md). It does not relicense real-person data,
generated experiment artifacts, prose research documentation, model bundles,
ZED/Isaac components, or other third-party assets.

## Intended use and limitations

This is research software, not a safety-certified perception system. Current
evidence is limited by synthetic imagery, a small number of scenes and
subjects, predominantly single-person trials, hardware-specific behavior, and
configuration-dependent depth failure modes. A method producing continuous
output under occlusion is not necessarily producing accurate output.

No DOI has been assigned to the initial repository release. For now, follow
[`CITATION.cff`](CITATION.cff) and include the exact repository commit used.
The software copyright notice is `Copyright 2026 Zhichao Huang`.

This public repository is a curated release. Private captures, identifiable
RGB-D data, generated experiment bundles, workstation-specific launchers,
device identifiers, and historical scripts containing machine-local settings
remain outside Git. Detailed frozen studies may therefore require separately
governed inputs that are not distributed here. See
[Data availability and privacy](DATA_AVAILABILITY.md).
