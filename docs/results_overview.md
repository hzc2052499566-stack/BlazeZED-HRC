# Results overview

This document was synchronized on 2026-10-05 against the completed studies and
local artifacts available through 2026-08-31. It is a map, not a replacement for
the frozen protocol, machine-readable summary, or detailed result file for an
experiment. The publication date is not the date of a new experimental run.

## How to interpret the table

- **Pass** means all stated gates in that frozen contract passed.
- **Mixed/negative** means at least one registered gate failed even if a useful
  secondary endpoint improved.
- **Engineering/exploratory** means feasibility or method development, not a
  confirmatory claim.
- A result is bounded by its tested scene, subject, resolution, frame rate,
  sensor mode, and endpoint.

## Baseline and component studies

| Study | Main observation | Overall status and boundary |
|---|---|---|
| Female static baseline, 10 s | 59 frames; detection and core-joint validity were both 100% | Completed baseline in one clean static synthetic scene; not a general detection rate |
| 640x360 distance scan | Full-image operating boundary was `[3.45 m, 3.50 m)` | Configuration-specific boundary |
| BlazePose Lite vs Full | Full favored offline accuracy; Lite favored live speed | Model choice is workload-specific |
| Depth sampling | `median_7x7` selected for clean offline depth; `wrist_aware_v5` selected for real live ZED | Cross-resolution/frame-rate transfer is not established |
| Kinematic constraints | K2g selected as the measured-joint default; K4 retained for optional inferred robustness | K4 provenance is separate and measured joints are never overwritten |

## Temporal tracking: useful effects, negative full contracts

The clean dynamic T2 method reduced the right-arm P95 step residual by
21.7%-30.7% over three runs. However, the run-by-run static-jitter gate failed
in `rep_02`; 7/8 gates passed and the overall registered contract failed.

A later 640x360 static replay passed its own 6/6 gates and reduced right-arm
3D jitter by 56.5%-68.0% at coverage 1.000. This is cross-resolution
generalization evidence, not a retroactive reversal of the dynamic result.

The fresh 960x600 static confirmatory study again reduced jitter, by
64.3%-77.1%, at coverage 1.000. It also increased mean position error by
16.1-17.9 mm and P95 error by 7.2-13.1 mm. The method was lower variance but
more biased; 6/8 gates passed and the full contract was negative.

## Rendered partial occlusion and K4

Fault-injection studies support K4's intended narrow use: it can bridge short
root-depth anomalies and up to 0.5 s of root-depth loss in the tested setup,
but not complete 2D shoulder loss.

The corrected rendered-occlusion pilot passed its manipulation checks. In the
three-run formal study, raw/K2 depth collapsed to the occluder surface while
K3/K4 preserved inferred output. Ten of eleven registered gates passed. The
failed replication gate expected full inferred coverage; observed K3/K4
occlusion coverage was `0.958 +/- 0.000`. The formal outcome remains mixed,
not a full pass.

The earlier v1 rendered-occlusion attempt produced no experimental data
because its occluder was attached under a camera child and did not render. Its
artifacts are retained as an excluded engineering record.

## Best-view selection and conditional streaming

The frozen measured-first View A/C formal study passed 5/5 gates. Under the
tested rendered occlusion, best-view measured coverage was
`1.000 +/- 0.000`, right-arm two-joint error was `19.45 +/- 0.64 mm`, and View C
was selected for all 180/180 active frames across the three runs.

The GT-free deployment-equivalence audit passed 7/7 gates: across 720 frames
and 1,440 joint outputs, it had zero selector-identity mismatches against the
formal selector and a maximum coordinate difference of `4.44e-16 m`.

Correct selection did not imply end-to-end real-time throughput. The always-on
dual CPU-worker decomposition reached only `17.08 +/- 0.39` paired FPS, and all
three runs failed the 27-FPS gate. The selector itself was not the bottleneck.

Conditional View C streaming v2 used four RGB-D pre-roll frames to remove the
onset K4 fallback seen with two-frame pre-roll. Its three-run repeatability v2
passed 25/25 per-run and 5/5 cross-run gates. Active measured coverage was
`1.000 +/- 0.000`, active mean/P95 error was
`20.22 +/- 0.48 / 25.87 +/- 0.56 mm`, and consumer rate was
`39.98 +/- 1.04 FPS`. That rate is a component/consumer measurement, not a
camera-to-output FPS claim.

## Real SVO and OBJ case study

The original SVO validity stage decoded 2,498 analysis frames and passed its
eight SVO gates. Camera-motion checks found an approximately 5-degree pose
change between frames 362 and 456, invalidating the planned single fixed-pose
registration assumption.

The selected SVO-to-OBJ registration route then failed four of five held-out
gates: held-out median depth difference was 102.26 mm, P95 was 1570.87 mm,
only 0.4938 of evaluated pixels were within 100 mm, and 0/24 validation frames
passed. The failure cannot distinguish a mismatched capture/session from a
registration optimum failure. In either case, the mesh is visualization-only
and **no clearance or separation value is valid**.

In the later same-subject real-ZED repeatability series, three runs provided
12,648 frozen frames and passed the implementation-lock/verifier layers. The
calibration profile qualified only 2/8 bones, so the registered main contract
was blocked and D2/K2g and D3/K4 were not run. D0 to D1 was a registered
negative: core measured coverage fell by 4.82-5.41 percentage points, and all
six wrist-recovery gates failed. This real replay result does not overturn the
separate 960x600/60 FPS live result; the configurations differ materially.

## `BODY_38` reference comparison

`BODY_38` was evaluated as a reference, never as baseline or GT. In the formal
Isaac comparison, all 8/8 validity gates passed in each run, but the registered
direction-replication gate R2 failed: the raw primary-endpoint direction was
`[A3, A2_v5, A3]`. The overall result is a registered negative.

The BlazePose rendered-depth anchor was reproducible (`SD/mean = 0.0148`),
whereas fitted `BODY_38` output was much less reproducible in the tested
configuration (`SD/mean = 0.3945`). Controlled replay showed the mechanism:
the same input and run-up reproduced exactly, but seeking into the stream
changed results substantially because the estimate depended on prior tracking
history. Consequently the study does not support a general claim that either
pipeline is more accurate.

Coverage is also not accuracy. In GT-free Track B, `BODY_38` continued to
return fitted skeleton output through occlusion; this non-abstention is not
proof that its occluded positions were correct.

## Adaptive camera and weighted limb length

The adaptive-camera line closed with a registered negative conclusion: its
objective worked as a usability filter, not an accuracy ranker. Visibility
gates passed, but rank correlation was `-0.39` and regret was 12.31 mm in the
frozen evaluation.

Weighted multi-view limb-length estimates materially reduced pooled bone mean
absolute error relative to equal weights and a fixed development view.
Nevertheless, both formal lines failed their registered robustness gate R5:
the original formal study exceeded the forearm MAD non-inferiority bound in
two runs, and the CTS5 follow-up exceeded the upper-arm MAD ratio in `rep_09`.
Both outcomes are mixed/negative rather than confirmatory passes.

CTS5 is a project method label, not a claimed standard acronym. It applies
per-view/per-bone bias calibration to scalar limb lengths and robustly combines
five views: trim the minimum and maximum with five or four eligible views,
use the median with three, and abstain with fewer than three. FS-CTS5 extends
that scheme to the bilateral upper arms, forearms, thighs, and shanks: eight
mapped major segments, not a literal full skeleton.

## FS-CTS5 20-bundle formal supplement

One synthetic female character was evaluated in four frozen scenarios, each
with five independent captures: clean motion, recoverable arm occlusion,
recoverable table occlusion, and severe table occlusion. Each capture exported
240 render steps from five cameras. All 20 bundles completed, providing 24,000
camera frames and 152,320 comparison rows. Captures, not individual frames, are
the independent units. All predictions were verified before the first formal
GT opening.

The primary contrasts were M0 (raw single primary view), M1 (raw available-view
uniform mean), M2 (bias-calibrated uniform mean), and M3 (FS-CTS5 robust fusion).
The gates were G1 accuracy, G2 output coverage, G3 per-segment P95
non-inferiority, G4 accurate-output rate, G5 severe-occlusion abstention, and
G6 recovery. G1/G2/G4/G5/G6 passed. G3 failed one clean right-thigh cell:

| M3 P95 | M2 P95 | Registered margin | Excess beyond allowed limit |
|---|---|---|---|
| 26.509 mm | 23.746 mm | 2.084 mm | 0.680 mm |

The complete registered contract is therefore **negative**, despite improved
average accuracy and other passed gates. Thresholds must not be widened after
observing this result. See the [aggregate gate summary](../results/fs_cts5_20_bundle_summary.json).

## FS-CTS5 cross-character industrial supplement

The first canonical same-input replay remains a preserved engineering failure:
worker 11 completed its payload but failed at directory promotion, later workers
were sequence-gated, exact QC was not run, and the selector was never called.
Its status remains `FAIL_sealed_replay_preserved_no_rerun`.

An independently frozen prospective recovery used a different write-once
destination. It completed 24/24 fresh workers with unique kernel process
identities and passed 12/12 same-input binary-exact sessions. The three selector
matrices contained 832,896 registered bits. The frozen selector was then called
exactly once and returned a registered negative: 0 of 6,126,120 candidate pairs
passed all hard gates in all three repeats. Coverage rejected 4,934,215 pairs and
recoverable-event replication rejected the remaining 1,191,905.

No view bank was authorized: `selected_view_ids`, `bank_view_ids`, and `m0_view_id`
are null. Confirmatory remains 0/12, excluded calibration remains 0/12, and formal
remains 0/36; overlay and G2 were not executed. This is a selector-level negative,
not a result for the planned 36-bundle method comparison. The old canonical
engineering failure and the new registered selector negative are separate records
and must not be pooled. See the
[terminal result](fs_cts5_common_bank_randomized_block_science_recovery_v3_terminal_results_cn.md).

## Cross-character successor probes

Successor-v3 repeated the same scientific boundary with an independent layout.
Its canonical raw capture completed 8/8, while the original long-path replay failed
before inference because of Windows `MAX_PATH`. A separately frozen short-path
recovery then completed 16/16 workers and passed 8/8 A/B exact-QC sessions. Its one
registered selector call audited 2,704,156 subsets and found zero qualifying subsets.
This is a selector-level scientific negative; the earlier canonical replay failure
remains a separate engineering record.

Successor-v4 `v4p1r6` has completed 12/12 fresh raw sessions and independent full
validation. Its replay-only authority was subsequently frozen on 2026-08-28
(`postcapture_replay_lock.json`, file SHA-256
`7067dbd68055071425b3bc43b7e6f5444846a451f30779becc3698d850ff3d6d`).
The lock permits a 24-worker A/B replay, but the derived directory contains no
worker output or replay closeout. Selector and formal authority remain false.
No selector, calibration, confirmatory or formal result exists for this cohort.
It is engineering progress and future-work material, not evidence needed to
complete the current report.

## Global claim boundaries

1. The baseline is BlazePose + ZED RGB-D.
2. Isaac USD Skeleton is simulation GT; `BODY_38` is only a reference estimate.
3. K4 inferred output is not measured coverage and never overwrites measured
   joints.
4. The right-arm two-joint endpoint is not full-body MPJPE.
5. Most core evidence is single-person and must not be generalized to
   validated multi-person operation.
6. Offline or component throughput is not camera-to-output FPS.
7. A failed overall gate remains failed even when a secondary endpoint improves.
8. Failed OBJ registration prohibits clearance claims.
9. A selector terminal failure does not authorize manual near-miss selection,
   threshold changes, a second selector call, or downstream capture.

## Evidence availability

This overview is the publication-cleared summary. The underlying frozen
protocols, manifests, captures, and generated bundles are not all distributed
in this repository because some contain machine-local identifiers or governed
human data. Their absence must not be interpreted as an open-data claim. See
[Data availability](../DATA_AVAILABILITY.md) and the
[Reproducibility guide](reproducibility.md).
