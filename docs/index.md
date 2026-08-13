# Documentation index

This page is the public entry point to the curated project documentation.
Detailed engineering logs, identifiable captures, generated experiment
bundles, and superseded protocols are retained outside this public repository.
The absence of those restricted artifacts does not change the status of a
reported positive, mixed, negative, aborted, or exploratory result.

## Public guides

- [Project README](../README.md) — scope, baseline identity, and repository map
- [Architecture](architecture.md) — processing chain, coordinate convention,
  provenance, and evaluation contracts
- [Results overview](results_overview.md) — consolidated positive, mixed,
  negative, and engineering-only findings
- [Reproducibility](reproducibility.md) — environment tiers, frozen inputs,
  replay rules, and validation workflow
- [Data availability](../DATA_AVAILABILITY.md) — release classes, privacy, and
  large-asset handling
- [Software licence scope](../LICENSE_SCOPE.md) — Apache-2.0 coverage and
  exclusions

## Method defaults

The project defaults are context-specific:

| Context | Frozen default |
|---|---|
| Clean offline static simulation | BlazePose Full, ROI 0.65, `median_7x7` |
| Offline dynamic GT replay | BlazePose Lite, ROI 0.65, warm-up 0 |
| Real ZED live | BlazePose Lite, ROI 0.65, `wrist_aware_v5` |
| Valid measured-joint constraint | K2g guarded eight-bone projection |
| Selected missing right-arm completion | K4 root-guarded inference, separate provenance |

These defaults are not interchangeable without a new validation. In
particular, pixel- and time-based `wrist_aware_v5` settings do not automatically
transfer between resolution/frame-rate regimes.

Ground-truth language is strict: the synchronized **Isaac USD Skeleton** is
simulation GT. ZED SDK `BODY_38` remains a comparison/reference estimate.

The consolidated [Results overview](results_overview.md) preserves the main
positive, mixed, negative, and engineering-only outcomes without publishing
machine-local protocol paths, device identifiers, or restricted human data.

## Reading status labels

- **Formal/confirmatory pass**: all gates in that frozen contract passed.
- **Mixed/negative**: useful secondary effects may exist, but at least one
  registered gate failed; the overall contract remains failed.
- **Pilot/engineering/exploratory**: method development or feasibility only.
- **Aborted/excluded**: not counted as a repeat and never silently replaced.
- **Draft**: a proposed protocol, not evidence.
