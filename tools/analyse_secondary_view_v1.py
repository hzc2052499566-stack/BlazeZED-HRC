"""Analyse a secondary-viewpoint partial-occlusion capture.

Reuses tools/analyse_partial_occlusion_v1.py unmodified so every viewpoint is
computed by identical code, then reinterprets the green-patch result with the
polarity a secondary view requires.

**Polarity warning.** In view A the manipulation check passes when the active
green fraction is high, because the occluder covers the forearm there. In a
secondary view the occluder is off to one side, so a high active green
fraction would mean that viewpoint is also blocked and multi-view is
pointless. The inherited `manipulation_check.checks` block is therefore
expected to report `all_passed = false` for a successful secondary-view
capture and must never be read as a failure. This tool writes an explicit
`secondary_view_unoccluded_check` block with the intended polarity.

Note that passing this check is necessary but not sufficient. The view B
pilot passed it and was still unusable, because the body self-occluded the
arm. Read the landmark eligibility rates as well: if the elbow and wrist
eligibility are near zero while the shoulder is near one, the viewpoint is
self-occluded regardless of the green fractions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_ANALYSER = ROOT / "tools" / "analyse_partial_occlusion_v1.py"
DEFAULT_SCENE_MANIFEST = (
    ROOT
    / "output"
    / "isaac_scenes"
    / "controlled_arm_motion_v6"
    / "controlled_arm_motion_v6_06_partial_occlusion_v2_manifest.json"
)
UNOCCLUDED_MAX_GREEN = 0.10
MIN_ENDPOINT_ELIGIBILITY = 0.50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base():
    spec = importlib.util.spec_from_file_location(
        "analyse_partial_occlusion_v1", BASE_ANALYSER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--view-label", required=True)
    parser.add_argument(
        "--scene-manifest", type=Path, default=DEFAULT_SCENE_MANIFEST
    )
    args = parser.parse_args()

    base = load_base()
    label = args.view_label.strip().lower()
    experiment_dir = args.experiment_dir.resolve()
    output = experiment_dir / f"partial_occlusion_view_{label}_report.json"
    if output.exists():
        raise base.AnalysisError(f"Refusing to overwrite: {output}")

    report = base.analyse(experiment_dir, args.scene_manifest.resolve())
    manipulation = report["manipulation_check"]
    landmarks = report["landmark_metrics"]

    unoccluded = {
        f"{phase}_{joint.split('_')[1]}_green_below_0p10": (
            manipulation[joint][phase]["mean_patch_green_fraction"]
            < UNOCCLUDED_MAX_GREEN
        )
        for joint in ("right_elbow", "right_wrist")
        for phase in ("active", "inactive")
    }
    unoccluded["all_passed"] = all(unoccluded.values())

    resolvable = {
        f"{joint}_eligibility_at_least_0p50": (
            landmarks[joint]["active"]["eligible_rate"]
            >= MIN_ENDPOINT_ELIGIBILITY
        )
        for joint in ("right_elbow", "right_wrist")
    }
    resolvable["all_passed"] = all(resolvable.values())

    report["purpose"] = (
        f"dynamic_partial_occlusion_view_{label}_v1_engineering_pilot"
    )
    report["claim_eligibility"] = "excluded_engineering_pilot_only"
    report["view"] = label
    report["secondary_view_unoccluded_check"] = {
        "threshold_max_green_fraction": UNOCCLUDED_MAX_GREEN,
        "meaning": (
            "The forearm must NOT be green in a secondary view, in either "
            "phase. This is the opposite polarity to the view A check."
        ),
        "checks": unoccluded,
    }
    report["secondary_view_resolvable_check"] = {
        "threshold_min_eligible_rate": MIN_ENDPOINT_ELIGIBILITY,
        "meaning": (
            "BlazePose must actually resolve the forearm from this "
            "viewpoint. The view B pilot passed the unoccluded check and "
            "failed here at 0.0000, because the body self-occluded the arm."
        ),
        "checks": resolvable,
    }
    report["base_analyser_sha256"] = sha256_file(BASE_ANALYSER)
    report["interpretation_rules"] = list(report["interpretation_rules"]) + [
        "POLARITY: manipulation_check.checks is inherited from the view A "
        "analyser and is EXPECTED to report all_passed = false here.",
        "A secondary view is usable only if BOTH "
        "secondary_view_unoccluded_check and "
        "secondary_view_resolvable_check pass.",
        "A single secondary-view pilot cannot support a repeatability claim.",
    ]
    base.atomic_write_json(output, report)

    print(f"Complete: {output}")
    print(f"  unoccluded check : {unoccluded['all_passed']}")
    print(f"  resolvable check : {resolvable['all_passed']}")
    for joint in ("right_shoulder", "right_elbow", "right_wrist"):
        block = landmarks[joint]["active"]
        print(
            "    {:16s} eligible {:.4f}  visibility {:.4f}".format(
                joint, block["eligible_rate"], block["visibility_mean"]
            )
        )
    detection = report["body_detection_rate"]
    print(
        "  detection  active {:.4f}  inactive {:.4f}".format(
            detection["active"], detection["inactive"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
