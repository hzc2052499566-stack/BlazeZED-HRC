"""Analyse the view B partial-occlusion capture.

Reuses tools/analyse_partial_occlusion_v1.py unmodified so view A and view B
are computed by identical code, then reinterprets the green-patch result with
the correct polarity for a second viewpoint.

**Polarity warning.** In view A the manipulation check passes when the active
green fraction is high, because the occluder covers the forearm there. In
view B the occluder is off to one side, so a high active green fraction would
mean the second viewpoint is *also* blocked and multi-view is pointless. The
inherited `manipulation_check.checks` block is therefore expected to report
`all_passed = false` for a successful view B capture, and must never be read
as a failure. This tool writes an explicit `view_b_unoccluded_check` block
with the intended polarity so no future reader has to remember that.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_ANALYSER = ROOT / "tools" / "analyse_partial_occlusion_v1.py"
OUTPUT_NAME = "partial_occlusion_view_b_report.json"
DEFAULT_SCENE_MANIFEST = (
    ROOT
    / "output"
    / "isaac_scenes"
    / "controlled_arm_motion_v6"
    / "controlled_arm_motion_v6_06_partial_occlusion_v2_manifest.json"
)
UNOCCLUDED_MAX_GREEN = 0.10


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
    parser.add_argument(
        "--scene-manifest", type=Path, default=DEFAULT_SCENE_MANIFEST
    )
    args = parser.parse_args()

    base = load_base()
    experiment_dir = args.experiment_dir.resolve()
    output = experiment_dir / OUTPUT_NAME
    if output.exists():
        raise base.AnalysisError(
            f"Refusing to overwrite view B report: {output}"
        )

    report = base.analyse(experiment_dir, args.scene_manifest.resolve())
    manipulation = report["manipulation_check"]
    checks = {
        "active_elbow_green_below_0p10": (
            manipulation["right_elbow"]["active"]["mean_patch_green_fraction"]
            < UNOCCLUDED_MAX_GREEN
        ),
        "active_wrist_green_below_0p10": (
            manipulation["right_wrist"]["active"]["mean_patch_green_fraction"]
            < UNOCCLUDED_MAX_GREEN
        ),
        "inactive_elbow_green_below_0p10": (
            manipulation["right_elbow"]["inactive"][
                "mean_patch_green_fraction"
            ]
            < UNOCCLUDED_MAX_GREEN
        ),
        "inactive_wrist_green_below_0p10": (
            manipulation["right_wrist"]["inactive"][
                "mean_patch_green_fraction"
            ]
            < UNOCCLUDED_MAX_GREEN
        ),
    }
    checks["all_passed"] = all(checks.values())

    report["purpose"] = "dynamic_partial_occlusion_view_b_v1_engineering_pilot"
    report["claim_eligibility"] = "excluded_engineering_pilot_only"
    report["view"] = "b"
    report["view_b_unoccluded_check"] = {
        "threshold_max_green_fraction": UNOCCLUDED_MAX_GREEN,
        "meaning": (
            "The forearm must NOT be green in view B, in either phase. This "
            "is the opposite polarity to the view A manipulation check."
        ),
        "checks": checks,
    }
    report["base_analyser_sha256"] = sha256_file(BASE_ANALYSER)
    report["interpretation_rules"] = list(report["interpretation_rules"]) + [
        "POLARITY: manipulation_check.checks is inherited from the view A "
        "analyser and is EXPECTED to report all_passed = false here. Read "
        "view_b_unoccluded_check instead.",
        "View B is a second viewpoint of the same manipulation, not a "
        "second experiment.",
        "A single view B pilot cannot support a repeatability claim.",
    ]
    base.atomic_write_json(output, report)

    print(f"Complete: {output}")
    print(
        "View B unoccluded check passed: "
        f"{checks['all_passed']}"
    )
    for joint in ("right_elbow", "right_wrist"):
        print(
            "  {:14s} active green {:.4f}  inactive green {:.4f}".format(
                joint,
                manipulation[joint]["active"]["mean_patch_green_fraction"],
                manipulation[joint]["inactive"]["mean_patch_green_fraction"],
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
