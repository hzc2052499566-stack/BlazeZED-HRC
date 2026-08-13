from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
TOOLS = WORKSPACE / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_view_ac_engineering_pair_v1 as pairing


JOINTS = tuple("joint_{:02d}".format(index) for index in range(15))


def rotation_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def rotation_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def world_point(sequence: int, joint_index: int) -> np.ndarray:
    return np.asarray(
        [
            1.6
            + 0.17 * math.sin(sequence / 19.0)
            + 0.031 * (joint_index % 5),
            -1.3
            + 0.14 * math.cos(sequence / 23.0)
            + 0.027 * (joint_index // 5),
            1.05 + 0.011 * joint_index + 0.0007 * sequence,
        ],
        dtype=np.float64,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_view(
    directory: Path,
    *,
    label: str,
    rotation_world_from_camera: np.ndarray,
    translation_world_from_camera: np.ndarray,
) -> None:
    directory.mkdir(parents=True)
    run_id = "synthetic_view_{}".format(label.lower())
    camera_prim = "/World/SyntheticView{}Camera".format(label)
    unit_hash = label.upper() * 64
    protocol = {
        "purpose": "deterministic_dynamic_rgbd_and_skeleton_gt",
        "run_id": run_id,
        "camera_prim_path": camera_prim,
        "time_codes_per_second": 60,
        "frames_per_cycle": 240,
        "cycle_count": 1,
        "sequence_count": 240,
        "excluded_closure_time_code": 240,
        "first_sequence": {
            "sample_index": 0,
            "sequence_index": 0,
            "cycle_index": 0,
            "animation_frame_code": 0,
            "usd_time_code": 0,
        },
        "last_sequence": {
            "sample_index": 239,
            "sequence_index": 239,
            "cycle_index": 0,
            "animation_frame_code": 239,
            "usd_time_code": 239,
        },
    }
    protocol_path = directory / "source_dynamic_rgbd_gt_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, sort_keys=True), encoding="utf-8"
    )
    protocol_hash = pairing.sha256_file(protocol_path)
    metadata = {
        "status": "complete_derived_unit_corrected",
        "sample_count": 240,
        "ground_truth_row_count": 3600,
        "unit": "operational_metre",
        "operational_metres_per_scene_unit": 1.0,
        "operational_unit_correction_factor": 100.0,
    }

    frame_directory = directory / "rgbd_frames"
    frame_directory.mkdir()
    manifest_rows: list[dict[str, object]] = []
    gt_rows: list[dict[str, object]] = []
    content_rows: list[dict[str, object]] = []
    for sequence in range(240):
        timeline = repr(sequence / 60.0)
        rgb_file = "rgb_{:04d}.npy".format(sequence)
        depth_file = "depth_{:04d}.npy".format(sequence)
        camera_params_file = "camera_params_{:04d}.json".format(sequence)
        (frame_directory / rgb_file).write_bytes(
            "rgb:{}:{}".format(label, sequence).encode("ascii")
        )
        (frame_directory / depth_file).write_bytes(
            "depth:{}:{}".format(label, sequence).encode("ascii")
        )
        (frame_directory / camera_params_file).write_text(
            json.dumps({"sequence_index": sequence}), encoding="utf-8"
        )
        manifest_rows.append(
            {
                "run_id": run_id,
                "sample_index": sequence,
                "sequence_index": sequence,
                "cycle_index": 0,
                "animation_frame_code": sequence,
                "usd_time_code": sequence,
                "timeline_time_s": timeline,
                "wall_time_ns": 1_000_000 + sequence,
                "camera_prim": camera_prim,
                "rgb_file": rgb_file,
                "depth_file": depth_file,
                "camera_params_file": camera_params_file,
                "width": 960,
                "height": 600,
                "fx": 370.8,
                "fy": 370.8,
                "cx": 480.0,
                "cy": 300.0,
                "depth_scale_to_m": 100,
                "camera_model": "pinhole",
                "protocol_sha256": protocol_hash,
                "source_depth_scale_to_m": 1,
                "operational_unit_correction_factor": 100,
                "unit_correction_protocol_sha256": unit_hash,
            }
        )
        for joint_index, joint in enumerate(JOINTS):
            world = world_point(sequence, joint_index)
            camera = rotation_world_from_camera.T @ (
                world - translation_world_from_camera
            )
            gt_rows.append(
                {
                    "run_id": run_id,
                    "sample_index": sequence,
                    "sequence_index": sequence,
                    "cycle_index": 0,
                    "animation_frame_code": sequence,
                    "usd_time_code": sequence,
                    "timeline_time_s": timeline,
                    "canonical_joint": joint,
                    "world_x_m": format(world[0], ".17g"),
                    "world_y_m": format(world[1], ".17g"),
                    "world_z_m": format(world[2], ".17g"),
                    "gt_x_m": format(camera[0], ".17g"),
                    "gt_y_m": format(camera[1], ".17g"),
                    "gt_z_m": format(camera[2], ".17g"),
                    "gt_semantics": "synchronized_per_frame",
                    "operational_unit_correction_factor": 100,
                    "unit_correction_protocol_sha256": unit_hash,
                }
            )
        content_rows.append(
            {
                "sample_index": sequence,
                "sequence_index": sequence,
                "cycle_index": 0,
                "animation_frame_code": sequence,
                "usd_time_code": sequence,
                "rgb_file": rgb_file,
                "depth_file": depth_file,
                "camera_params_file": camera_params_file,
                "rgb_content_sha256": pairing.sha256_file(
                    frame_directory / rgb_file
                ),
                "depth_content_sha256": pairing.sha256_file(
                    frame_directory / depth_file
                ),
                "camera_params_content_sha256": pairing.sha256_file(
                    frame_directory / camera_params_file
                ),
                "gt_coordinates_sha256": "d" * 64,
                "frame_content_sha256": "e" * 64,
                "hash_schema": pairing.SOURCE_CONTENT_HASH_SCHEMA,
            }
        )
    write_csv(directory / "rgbd_manifest.csv", manifest_rows)
    write_csv(directory / "ground_truth_joints.csv", gt_rows)
    write_csv(directory / "source_frame_content_hashes.csv", content_rows)
    (directory / "ground_truth_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    state = {
        "status": "complete",
        "correction_factor": 100.0,
        "source_sample_count": 240,
        "source_gt_row_count": 3600,
        "hard_linked_rgb_depth_file_count": 480,
        "rewritten_camera_params_file_count": 240,
        "output_sha256": {
            name: pairing.sha256_file(directory / name)
            for name in pairing.STATE_OUTPUT_FILES
        },
    }
    (directory / "unit_correction_state.json").write_text(
        json.dumps(state, sort_keys=True), encoding="utf-8"
    )


def rewrite_csv_cell(
    path: Path,
    row_index: int,
    field: str,
    value: object,
) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[row_index][field] = value
    write_csv(path, rows)


def refresh_registered_hash(directory: Path, name: str) -> None:
    state_path = directory / "unit_correction_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["output_sha256"][name] = pairing.sha256_file(directory / name)
    state_path.write_text(
        json.dumps(state, sort_keys=True), encoding="utf-8"
    )


class ViewACEngineeringPairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rotation_a = rotation_z(17.0) @ rotation_x(-4.0)
        self.translation_a = np.asarray([0.4, -0.8, 1.3])
        self.rotation_c = rotation_z(-31.0) @ rotation_x(7.0)
        self.translation_c = np.asarray([2.7, 0.5, 1.45])

    def create_inputs(self, root: Path) -> tuple[Path, Path]:
        view_a = root / "view_a"
        view_c = root / "view_c"
        make_view(
            view_a,
            label="A",
            rotation_world_from_camera=self.rotation_a,
            translation_world_from_camera=self.translation_a,
        )
        make_view(
            view_c,
            label="C",
            rotation_world_from_camera=self.rotation_c,
            translation_world_from_camera=self.translation_c,
        )
        return view_a, view_c

    def test_builds_exact_pair_and_recovers_column_vector_transforms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            view_a, view_c = self.create_inputs(root)
            output = root / "engineering_pair"

            result = pairing.build_pair(view_a, view_c, output)

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["frame_count"], 240)
            self.assertEqual(result["correspondence_count"], 3600)
            self.assertEqual(
                result["claim_eligibility"],
                "excluded_engineering_replay_only",
            )
            self.assertEqual(
                result["calibration_source"],
                "gt_correspondence_engineering_only",
            )
            self.assertFalse(result["eligible_for_formal"])

            sync = json.loads(
                (output / "sync_manifest.json").read_text(encoding="utf-8")
            )
            calibration = json.loads(
                (output / "calibration.json").read_text(encoding="utf-8")
            )
            for payload in (sync, calibration):
                self.assertEqual(
                    payload["claim_eligibility"],
                    "excluded_engineering_replay_only",
                )
                self.assertEqual(
                    payload["calibration_source"],
                    "gt_correspondence_engineering_only",
                )
                self.assertFalse(payload["eligible_for_formal"])
                self.assertTrue(payload["uses_ground_truth"])
            expected_audit_files = {
                "source_frame_content_hashes.csv",
                "source_dynamic_rgbd_gt_protocol.json",
            }
            for view in ("a", "c"):
                self.assertTrue(
                    expected_audit_files.issubset(
                        sync["views"][view]["source_sha256"]
                    )
                )
                self.assertTrue(
                    expected_audit_files.issubset(
                        calibration["source_sha256"]["view_{}".format(view)]
                    )
                )
            self.assertEqual(len(sync["pairs"]), 240)
            self.assertFalse(
                sync["synchronization_contract"]["same_render_step"]
            )
            self.assertFalse(
                sync["synchronization_contract"]["wall_time_used_for_pairing"]
            )

            world_a = calibration["views"]["a"]["world_from_camera"]
            world_c = calibration["views"]["c"]["world_from_camera"]
            np.testing.assert_allclose(
                world_a["rotation_3x3"], self.rotation_a, atol=1.0e-12
            )
            np.testing.assert_allclose(
                world_a["translation_m"], self.translation_a, atol=1.0e-12
            )
            np.testing.assert_allclose(
                world_c["rotation_3x3"], self.rotation_c, atol=1.0e-12
            )
            np.testing.assert_allclose(
                world_c["translation_m"], self.translation_c, atol=1.0e-12
            )
            expected_a_from_c_rotation = self.rotation_a.T @ self.rotation_c
            expected_a_from_c_translation = self.rotation_a.T @ (
                self.translation_c - self.translation_a
            )
            a_from_c = calibration["relative_transforms"]["a_from_c"]
            np.testing.assert_allclose(
                a_from_c["rotation_3x3"],
                expected_a_from_c_rotation,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                a_from_c["translation_m"],
                expected_a_from_c_translation,
                atol=1.0e-12,
            )
            self.assertEqual(
                a_from_c["homogeneous_4x4"][3], [0.0, 0.0, 0.0, 1.0]
            )
            self.assertAlmostEqual(
                calibration["baseline_m"],
                float(np.linalg.norm(self.translation_c - self.translation_a)),
                places=12,
            )
            self.assertLess(
                calibration["cross_view_fit"]["a_from_c"][
                    "maximum_residual_m"
                ],
                1.0e-10,
            )

    def test_rejects_one_world_gt_change_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            view_a, view_c = self.create_inputs(root)
            rewrite_csv_cell(
                view_c / "ground_truth_joints.csv",
                117,
                "world_x_m",
                "9.25",
            )
            refresh_registered_hash(view_c, "ground_truth_joints.csv")
            output = root / "must_not_exist"

            with self.assertRaisesRegex(
                pairing.EngineeringPairingError,
                "operational-world GT mismatch",
            ):
                pairing.build_pair(view_a, view_c, output)

            self.assertFalse(output.exists())

    def test_rejects_nonexact_timecode_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            view_a, view_c = self.create_inputs(root)
            rewrite_csv_cell(
                view_c / "rgbd_manifest.csv",
                73,
                "usd_time_code",
                "72",
            )
            refresh_registered_hash(view_c, "rgbd_manifest.csv")
            output = root / "must_not_exist"

            with self.assertRaisesRegex(
                pairing.EngineeringPairingError,
                "exact animation/USD time code",
            ):
                pairing.build_pair(view_a, view_c, output)

            self.assertFalse(output.exists())

    def test_rejects_registered_output_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            view_a, view_c = self.create_inputs(root)
            rewrite_csv_cell(
                view_c / "rgbd_manifest.csv",
                5,
                "wall_time_ns",
                "999999999",
            )
            output = root / "must_not_exist"

            with self.assertRaisesRegex(
                pairing.EngineeringPairingError,
                "registered output hash mismatch for rgbd_manifest.csv",
            ):
                pairing.build_pair(view_a, view_c, output)

            self.assertFalse(output.exists())

    def test_rejects_missing_manifest_referenced_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            view_a, view_c = self.create_inputs(root)
            (view_c / "rgbd_frames" / "depth_0041.npy").unlink()
            output = root / "must_not_exist"

            with self.assertRaisesRegex(
                pairing.EngineeringPairingError,
                "manifest references missing depth_file",
            ):
                pairing.build_pair(view_a, view_c, output)

            self.assertFalse(output.exists())

    def test_rejects_source_content_manifest_linkage_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            view_a, view_c = self.create_inputs(root)
            rewrite_csv_cell(
                view_c / "source_frame_content_hashes.csv",
                12,
                "rgb_file",
                "rgb_wrong.npy",
            )
            refresh_registered_hash(
                view_c, "source_frame_content_hashes.csv"
            )
            output = root / "must_not_exist"

            with self.assertRaisesRegex(
                pairing.EngineeringPairingError,
                "source content/manifest mismatch",
            ):
                pairing.build_pair(view_a, view_c, output)

            self.assertFalse(output.exists())

    def test_rejects_preserved_protocol_not_identified_by_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            view_a, view_c = self.create_inputs(root)
            protocol_path = (
                view_c / "source_dynamic_rgbd_gt_protocol.json"
            )
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol["camera_prim_path"] = "/World/TamperedCamera"
            protocol_path.write_text(
                json.dumps(protocol, sort_keys=True), encoding="utf-8"
            )
            refresh_registered_hash(
                view_c, "source_dynamic_rgbd_gt_protocol.json"
            )
            output = root / "must_not_exist"

            with self.assertRaisesRegex(
                pairing.EngineeringPairingError,
                "manifest protocol hash does not identify",
            ):
                pairing.build_pair(view_a, view_c, output)

            self.assertFalse(output.exists())

    def test_refuses_existing_output_without_touching_sentinel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            view_a, view_c = self.create_inputs(root)
            output = root / "existing"
            output.mkdir()
            sentinel = output / "user_owned.txt"
            sentinel.write_text("preserve", encoding="utf-8")

            with self.assertRaisesRegex(
                pairing.EngineeringPairingError,
                "Refusing to overwrite",
            ):
                pairing.build_pair(view_a, view_c, output)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                ["user_owned.txt"],
            )

    def test_rejects_output_inside_a_preserved_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            view_a, view_c = self.create_inputs(root)

            with self.assertRaisesRegex(
                pairing.EngineeringPairingError,
                "must not be inside",
            ):
                pairing.build_pair(view_a, view_c, view_a / "derived")


if __name__ == "__main__":
    unittest.main()
