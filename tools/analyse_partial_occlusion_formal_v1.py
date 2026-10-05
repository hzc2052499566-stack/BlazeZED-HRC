"""Per-repeat analysis wrapper for the partial-occlusion formal stage.

Reuses tools/analyse_partial_occlusion_v1.py unmodified so the formal repeats
and the v2 Pilot are computed by exactly the same code.  Only the report file
name and the recorded claim class differ, keeping rule 9.6 satisfied: an
engineering Pilot report and a formal repeat report never share a filename.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_ANALYSER = ROOT / "tools" / "analyse_partial_occlusion_v1.py"
OUTPUT_NAME = "partial_occlusion_formal_report.json"
DEFAULT_SCENE_MANIFEST = (
    ROOT
    / "output"
    / "isaac_scenes"
    / "controlled_arm_motion_v6"
    / "controlled_arm_motion_v6_06_partial_occlusion_v2_manifest.json"
)


def load_base():
    spec = importlib.util.spec_from_file_location(
        "analyse_partial_occlusion_v1",
        BASE_ANALYSER,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument(
        "--scene-manifest",
        type=Path,
        default=DEFAULT_SCENE_MANIFEST,
    )
    parser.add_argument("--repeat-index", type=int, required=True)
    args = parser.parse_args()

    base = load_base()
    experiment_dir = args.experiment_dir.resolve()
    output = experiment_dir / OUTPUT_NAME
    if output.exists():
        raise base.AnalysisError(
            f"Refusing to overwrite formal report: {output}"
        )
    report = base.analyse(experiment_dir, args.scene_manifest.resolve())
    report["purpose"] = "dynamic_partial_occlusion_formal_v1"
    report["claim_eligibility"] = "formal_repeat_pending_aggregation"
    report["repeat_index"] = int(args.repeat_index)
    report["base_analyser_sha256"] = base.__dict__.get(
        "__analyser_sha256__",
        _sha256(BASE_ANALYSER),
    )
    report["interpretation_rules"] = list(report["interpretation_rules"]) + [
        "A single formal repeat is not the reporting unit; only the "
        "three-run aggregate is.",
    ]
    base.atomic_write_json(output, report)
    print(f"Complete: {output}")
    print(
        "Manipulation check passed: "
        f"{report['manipulation_check']['checks']['all_passed']}"
    )
    return 0


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
