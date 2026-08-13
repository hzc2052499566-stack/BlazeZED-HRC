"""GT-free adaptive camera positioning objective.

No file I/O and no ground-truth interface.  Joint positions and the occluder
model are supplied by the caller, so the same code path serves every
information tier and the module cannot tell one from another:

    Tier-O   caller passes Skeleton GT joints and the exact occluder box.
             An upper bound.  Never report it as achievable performance.
    Tier-D   caller passes View A measured-first estimates and an occluder
             proxy built from View A's own depth map.  Deployable.
    Tier-M   caller passes View A estimates and FREE_SPACE.  Ablation: how
             much does knowing about the occluder buy?

Feasibility is four gates; ranking is one continuous term.  Every threshold
is inherited from a frozen result and none is fitted here.

    F1 in_roi        the four registered joints project inside the ROI 0.65
                     window (agents.md 3.3)
    F2 distance      registration-time only, see radius_within_envelope
    F3 unoccluded    the segment from the eye to each arm joint must not
                     intersect the occluder
    F4 resolvable    wrist-to-torso-axis separation above 20 px (agents.md
                     5.16 MIN_SEPARATION_PX)

    S  ranking       wrist-to-torso-axis separation, px

F2 is not a per-frame gate.  agents.md 5.3 swept a static T-pose at fixed
camera distances, so the envelope describes where a camera may be installed
relative to the subject's nominal standing position; the subject's own motion
around that position was never what the sweep measured.  Gating the per-frame
pelvis range against it also lets the surface-versus-pivot offset of
agents.md 4.2 act as a hidden threshold -- ground-truth pivots and estimated
surface landmarks sit about 0.14 m apart, enough to pass one tier and reject
another on identical geometry.  ``pelvis_range_m`` is therefore recorded per
frame and never scored.

Three decisions differ from scan_camera_placements_v1 and all are narrowings.

F4 and S use the wrist alone.  The 20 px threshold was calibrated on the
wrist, and elbow separation runs 18-20 px lower at the same viewpoint, so
scoring the elbow against it would use the threshold outside its
calibration -- the failure mode agents.md 4.11 records for the occlusion
depth margin.  Elbow separation is recorded, not scored.

F3 carries no margin.  The exact oriented box reproduces all three measured
outcomes unaided: view A blocked at both arm joints, the 0.12 m stereo
baseline clearing the elbow by about 5 mm while the wrist still intersects,
and view C clear by more than a metre.  The per-joint edge insets of v1 were
calibration constants for a disc approximation of the cuboid and do not
carry over to exact geometry.

The candidate camera geometry (arc centre, radius, aim point) is a
registration-time constant: it describes where the cameras are bolted, which
an offline survey may legitimately inform.  Only the per-frame objective is
required to be GT-free.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


METHOD_VERSION = "adaptive_camera_objective_v1b_20260807"
MEASURED_SOURCE = "k2_reliable_measured"

# agents.md 3.3 live formal configuration.
ROI_SCALE = 0.65
# agents.md 5.16.  Calibrated on the wrist; see the module docstring.
MIN_WRIST_SEPARATION_PX = 20.0
# agents.md 5.3 distance sweep plus the 3.3 live configuration.
DISTANCE_ENVELOPE_M = (2.00, 3.50)
# Frozen in multiview_best_view_measured_first_v1; reused verbatim.
OCCLUSION_DEPTH_MARGIN_M = 0.50

ARM_JOINTS = ("right_elbow", "right_wrist")
TORSO_JOINTS = ("right_shoulder", "pelvis")
REQUIRED_JOINTS = TORSO_JOINTS + ARM_JOINTS

WORLD_UP = np.array([0.0, 0.0, 1.0])
# Distance to a convex set is convex along a segment, so the minimum is
# unimodal.  32 ternary steps shrink a 3.6 m segment to about 30 um, far below
# anything the objective resolves, and the count is fixed so there is no
# tolerance-dependent branch.
_TERNARY_ITERATIONS = 32


class ObjectiveError(RuntimeError):
    """Raised when the objective is called with an incomplete contract."""


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------


class LookAtCamera:
    """A ZED-convention camera: +X forward, +Y left, +Z up (agents.md 4.3).

    ``basis`` rows are the camera axes expressed in world coordinates, so
    ``basis @ (p - eye)`` yields ZED camera coordinates.
    """

    __slots__ = ("eye", "basis", "fx", "fy", "cx", "cy", "width", "height")

    def __init__(self, eye, basis, fx: float, fy: float, cx: float, cy: float,
                 width: int, height: int) -> None:
        self.eye = np.asarray(eye, dtype=float)
        self.basis = np.asarray(basis, dtype=float)
        if self.eye.shape != (3,) or self.basis.shape != (3, 3):
            raise ObjectiveError("Camera eye must be 3-vector and basis 3x3.")
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.width, self.height = int(width), int(height)

    @classmethod
    def look_at(cls, eye, target, fx: float, fy: float, cx: float, cy: float,
                width: int, height: int) -> "LookAtCamera":
        eye = np.asarray(eye, dtype=float)
        forward = np.asarray(target, dtype=float) - eye
        norm = float(np.linalg.norm(forward))
        if norm < 1.0e-9:
            raise ObjectiveError("Camera target coincides with the eye.")
        forward = forward / norm
        left = np.cross(WORLD_UP, forward)
        norm = float(np.linalg.norm(left))
        if norm < 1.0e-9:
            raise ObjectiveError("Camera axis is degenerate (looking along world up).")
        left = left / norm
        return cls(eye, np.stack([forward, left, np.cross(forward, left)]),
                   fx, fy, cx, cy, width, height)

    @classmethod
    def from_extrinsics(cls, rotation_3x3, translation_m, fx: float, fy: float,
                        cx: float, cy: float, width: int, height: int) -> "LookAtCamera":
        """Build from a ``world_from_camera_zed`` block (columns are axes)."""
        return cls(translation_m, np.asarray(rotation_3x3, dtype=float).T,
                   fx, fy, cx, cy, width, height)

    def camera_point(self, point_world) -> np.ndarray:
        return self.basis @ (np.asarray(point_world, dtype=float) - self.eye)

    def project(self, point_world) -> tuple[np.ndarray, float] | None:
        """Return ((u, v), forward_m) or None when the point is behind."""
        v = self.camera_point(point_world)
        if v[0] <= 1.0e-9:
            return None
        return (np.array([self.cx - self.fx * v[1] / v[0],
                          self.cy - self.fy * v[2] / v[0]]), float(v[0]))

    def inside_roi(self, pixel) -> bool:
        half_w = self.width * ROI_SCALE / 2.0
        half_h = self.height * ROI_SCALE / 2.0
        return (abs(float(pixel[0]) - self.cx) <= half_w
                and abs(float(pixel[1]) - self.cy) <= half_h)


# --------------------------------------------------------------------------
# occluder models
# --------------------------------------------------------------------------


class FreeSpace:
    """Tier-M: nothing ever blocks."""

    kind = "free_space"

    def clearance_m(self, eye, joint) -> float:
        return math.inf


FREE_SPACE = FreeSpace()


class OrientedBox:
    """Tier-O: the exact registered cuboid.

    ``axes`` rows are unit vectors; ``half_extents`` are the corresponding
    half sizes.  Distance from a point to a convex set is convex, and the
    segment is a line, so the distance along the segment is unimodal and a
    ternary search finds the minimum without any tolerance-dependent branch.
    """

    kind = "oriented_box"

    __slots__ = ("centre", "axes", "half_extents")

    def __init__(self, centre, axes, half_extents) -> None:
        self.centre = np.asarray(centre, dtype=float)
        self.axes = np.asarray(axes, dtype=float)
        self.half_extents = np.asarray(half_extents, dtype=float)
        if (self.centre.shape != (3,) or self.axes.shape != (3, 3)
                or self.half_extents.shape != (3,)):
            raise ObjectiveError("Oriented box requires centre 3, axes 3x3, half 3.")

    @classmethod
    def from_usd_matrix(cls, matrix_4x4) -> "OrientedBox":
        """USD row-major: rows 0..2 are scaled basis vectors, row 3 translates.

        A ``Cube`` with ``size = 2`` has unit half extents, so each row length
        is that axis's half extent in world metres.
        """
        matrix = np.asarray(matrix_4x4, dtype=float)
        if matrix.shape != (4, 4):
            raise ObjectiveError("USD transform must be 4x4.")
        scaled = matrix[:3, :3]
        half = np.linalg.norm(scaled, axis=1)
        if float(half.min()) < 1.0e-9:
            raise ObjectiveError("Degenerate occluder axis.")
        return cls(matrix[3, :3], scaled / half[:, None], half)

    def _point_distances(self, points) -> np.ndarray:
        """Distance from each row of ``points`` to the box surface, 0 inside."""
        local = np.atleast_2d(np.asarray(points, dtype=float)) - self.centre
        local = local @ self.axes.T
        return np.linalg.norm(np.maximum(np.abs(local) - self.half_extents, 0.0), axis=1)

    def clearance_m(self, eye, joint) -> float:
        a = np.asarray(eye, dtype=float)
        delta = np.asarray(joint, dtype=float) - a
        low, high = 0.0, 1.0
        for _ in range(_TERNARY_ITERATIONS):
            first = low + (high - low) / 3.0
            second = high - (high - low) / 3.0
            probes = self._point_distances(np.stack([a + delta * first, a + delta * second]))
            if probes[0] < probes[1]:
                high = second
            else:
                low = first
        # The bracket endpoints are kept as well as its midpoint: when the
        # minimum sits at t = 0 or t = 1 the bracket never closes on it, and
        # the midpoint alone is off by half the remaining width.
        samples = np.stack([a + delta * t for t in (low, (low + high) / 2.0, high)])
        return float(self._point_distances(samples).min())


class DepthSurfaceProxy:
    """Tier-D: the occluding surface as View A actually observed it.

    Each retained pixel becomes a sphere of half its own pixel footprint, so
    the sampling radius follows from the intrinsics rather than from a fitted
    constant.  The proxy is a *front surface only* -- View A cannot see the
    back or the sides of whatever is blocking it -- so a candidate ray may
    pass behind the real obstacle and be scored clear.  That gap is the
    intended content of the Tier-D minus Tier-O comparison, not a defect to
    be patched with an invented depth.
    """

    kind = "depth_surface_proxy"

    __slots__ = ("points", "radii")

    def __init__(self, points, radii) -> None:
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.radii = np.asarray(radii, dtype=float).reshape(-1)
        if self.points.shape[0] != self.radii.shape[0]:
            raise ObjectiveError("Proxy points and radii must match in length.")

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def clearance_m(self, eye, joint) -> float:
        if self.points.shape[0] == 0:
            return math.inf
        a = np.asarray(eye, dtype=float)
        delta = np.asarray(joint, dtype=float) - a
        denominator = float(delta @ delta)
        if denominator < 1.0e-18:
            raise ObjectiveError("Degenerate eye-to-joint segment.")
        t = np.clip((self.points - a) @ delta / denominator, 0.0, 1.0)
        closest = a + t[:, None] * delta
        distance = np.linalg.norm(self.points - closest, axis=1) - self.radii
        return max(float(distance.min()), 0.0)


def build_depth_surface_proxy(
    depth_m: np.ndarray,
    camera: LookAtCamera,
    seed_pixels: Iterable[Sequence[float]] | None,
    torso_reference_depth_m: float,
) -> DepthSurfaceProxy:
    """Lift the near surface out of a depth map, GT-free.

    A pixel joins the surface when it is closer than the torso reference by
    more than ``OCCLUSION_DEPTH_MARGIN_M`` -- the same frozen criterion the
    deployed selector already uses to call a joint depth unreliable.  No new
    threshold is introduced.

    ``seed_pixels`` selects between two specifications.  A list of pixels
    flood fills only the connected component reached from those seeds, which
    keeps the proxy tight but inherits the reliability of the 2D landmarks
    used as seeds; pass the whole arm chain through ``rasterise_polyline``
    rather than two individual landmarks.  ``None`` keeps every near pixel in
    the frame, which needs no landmark at all but is not selective for
    occluders: the A1 scan measured 105,600 retained points -- the floor -- in
    frames where nothing was occluding at all.
    """
    if depth_m.ndim != 2:
        raise ObjectiveError("Depth map must be a 2D array in metres.")
    height, width = depth_m.shape
    finite = np.isfinite(depth_m)
    near = finite & ((torso_reference_depth_m - depth_m) > OCCLUSION_DEPTH_MARGIN_M)
    if not near.any():
        return DepthSurfaceProxy(np.zeros((0, 3)), np.zeros(0))

    if seed_pixels is None:
        row_index, column_index = np.nonzero(near)
    else:
        visited = np.zeros_like(near, dtype=bool)
        stack: list[tuple[int, int]] = []
        for pixel in seed_pixels:
            column, row = int(round(float(pixel[0]))), int(round(float(pixel[1])))
            if 0 <= row < height and 0 <= column < width and near[row, column]:
                stack.append((row, column))
        kept: list[tuple[int, int]] = []
        while stack:
            row, column = stack.pop()
            if visited[row, column]:
                continue
            visited[row, column] = True
            kept.append((row, column))
            for next_row, next_column in ((row - 1, column), (row + 1, column),
                                          (row, column - 1), (row, column + 1)):
                if (0 <= next_row < height and 0 <= next_column < width
                        and near[next_row, next_column] and not visited[next_row, next_column]):
                    stack.append((next_row, next_column))
        if not kept:
            return DepthSurfaceProxy(np.zeros((0, 3)), np.zeros(0))
        row_index = np.array([item[0] for item in kept])
        column_index = np.array([item[1] for item in kept])

    rows = row_index.astype(float)
    columns = column_index.astype(float)
    depth = depth_m[row_index, column_index].astype(float)
    # agents.md 3.1 back-projection, then camera -> world.
    camera_points = np.stack([
        depth,
        -(columns - camera.cx) * depth / camera.fx,
        -(rows - camera.cy) * depth / camera.fy,
    ], axis=1)
    world = camera_points @ camera.basis + camera.eye
    return DepthSurfaceProxy(world, depth / camera.fx / 2.0)


# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------


def torso_separation_px(shoulder_px, pelvis_px, joint_px) -> float:
    """Perpendicular pixel distance from a joint to the torso axis."""
    axis = np.asarray(pelvis_px, dtype=float) - np.asarray(shoulder_px, dtype=float)
    length = float(np.linalg.norm(axis))
    if length < 1.0e-9:
        raise ObjectiveError("Torso axis is degenerate in the image plane.")
    axis = axis / length
    delta = np.asarray(joint_px, dtype=float) - np.asarray(shoulder_px, dtype=float)
    return float(np.linalg.norm(delta - axis * float(delta @ axis)))


def evaluate_candidate(
    camera: LookAtCamera,
    joints_world: Mapping[str, Sequence[float] | None],
    occluder: Any = FREE_SPACE,
    reference_camera: LookAtCamera | None = None,
) -> dict[str, Any]:
    """Score one candidate viewpoint for one frame.

    ``joints_world`` must carry the four registered joints; a ``None`` entry
    means the upstream estimate was unavailable and the candidate cannot be
    scored.  Returns feasibility, the ranking score and the record-only
    quantities.  ``score`` is ``-inf`` for an infeasible candidate.
    """
    missing = [name for name in REQUIRED_JOINTS if name not in joints_world]
    if missing:
        raise ObjectiveError("Missing registered joints: {}".format(missing))

    result: dict[str, Any] = {
        "feasible": False,
        "reason": "",
        "score": -math.inf,
        "wrist_separation_px": None,
        "elbow_separation_px": None,
        "pelvis_range_m": None,
        "clearance_m": {},
        "triangulation_angle_deg": None,
        "baseline_m": None,
        "method_version": METHOD_VERSION,
    }

    unavailable = [name for name in REQUIRED_JOINTS if joints_world.get(name) is None]
    if unavailable:
        result["reason"] = "upstream_estimate_unavailable"
        return result

    projected: dict[str, np.ndarray] = {}
    for name in REQUIRED_JOINTS:
        item = camera.project(joints_world[name])
        if item is None:
            result["reason"] = "behind_camera"
            return result
        projected[name] = item[0]
        if name == "pelvis":
            result["pelvis_range_m"] = float(
                np.linalg.norm(np.asarray(joints_world[name], dtype=float) - camera.eye))

    # Record-only geometry, computed before the gates so a rejected candidate
    # is still fully described.
    result["wrist_separation_px"] = torso_separation_px(
        projected["right_shoulder"], projected["pelvis"], projected["right_wrist"])
    result["elbow_separation_px"] = torso_separation_px(
        projected["right_shoulder"], projected["pelvis"], projected["right_elbow"])
    if reference_camera is not None:
        result["baseline_m"] = float(np.linalg.norm(camera.eye - reference_camera.eye))
        result["triangulation_angle_deg"] = triangulation_angle_deg(
            camera.eye, reference_camera.eye, joints_world["right_wrist"])

    # F1
    if not all(camera.inside_roi(pixel) for pixel in projected.values()):
        result["reason"] = "outside_roi"
        return result
    # F2 is a registration-time check; see radius_within_envelope.
    # F3
    for name in ARM_JOINTS:
        clearance = occluder.clearance_m(camera.eye, joints_world[name])
        result["clearance_m"][name] = None if math.isinf(clearance) else float(clearance)
    if any(result["clearance_m"][name] is not None and result["clearance_m"][name] <= 0.0
           for name in ARM_JOINTS):
        result["reason"] = "occluded"
        return result
    # F4
    if result["wrist_separation_px"] <= MIN_WRIST_SEPARATION_PX:
        result["reason"] = "self_occluded"
        return result

    result["feasible"] = True
    result["reason"] = "feasible"
    result["score"] = float(result["wrist_separation_px"])
    return result


class MeasuredGeometryHold:
    """Feed the objective only geometry that came from a measurement.

    The frozen selector already labels each arm joint ``k2_reliable_measured``
    or ``k4_fallback_inferred``.  Letting an inferred joint drive the objective
    was measured to break it: in exactly the occluded regime where the
    fallback fires, View A's two-joint error is 191.82 +/- 4.11 mm
    (agents.md 5.24), which at 3.5 m projects to roughly 20 px of apparent
    torso separation -- the same size as the whole 20 px threshold.  The A1
    scan found this pushed a viewpoint whose true wrist separation is 4.82 px
    over the threshold in 46 of 60 occluded frames.

    So the objective abstains on inferred geometry and reuses the last frame
    whose registered joints were all measured.  This suits the frozen
    conditional architecture, which already needs lead time -- agents.md 5.29
    sends four frames of RGB-D pre-roll before a burst.  The cost is that the
    viewpoint cannot be re-planned *during* an occlusion.

    No maximum hold age is imposed.  ``age_frames`` is reported so that a cap
    can be registered later against evidence rather than guessed now.
    """

    __slots__ = ("_joints", "_age")

    def __init__(self) -> None:
        self._joints: dict[str, Any] | None = None
        self._age = 0

    def update(self, joints_world: Mapping[str, Any],
               arm_sources: Mapping[str, str]) -> dict[str, Any]:
        complete = all(joints_world.get(name) is not None for name in REQUIRED_JOINTS)
        measured = all(arm_sources.get(name) == MEASURED_SOURCE for name in ARM_JOINTS)
        if complete and measured:
            self._joints = dict(joints_world)
            self._age = 0
            return {"joints": self._joints, "available": True, "held": False,
                    "age_frames": 0, "reason": "measured"}
        if self._joints is None:
            return {"joints": None, "available": False, "held": False,
                    "age_frames": None, "reason": "no_measured_frame_yet"}
        self._age += 1
        return {"joints": self._joints, "available": True, "held": True,
                "age_frames": self._age,
                "reason": "held_incomplete" if not complete else "held_inferred"}


def rasterise_polyline(pixels: Sequence[Sequence[float]],
                       step_px: float = 1.0) -> list[np.ndarray]:
    """Sample a pixel polyline at one-pixel spacing.

    Used to seed the occluder proxy from the whole shoulder-elbow-wrist chain
    instead of two individual landmarks.  A single landmark drifts under
    occlusion -- agents.md 5.13 attributes the entire coverage loss to 2D --
    and the A1 scan found the seeds landing on background or on the torso in
    28 of 60 occluded frames.  The step is the pixel grid itself, not a
    fitted value.
    """
    points = [np.asarray(pixel, dtype=float) for pixel in pixels]
    if len(points) < 2:
        return list(points)
    samples: list[np.ndarray] = []
    for start, end in zip(points, points[1:]):
        span = end - start
        length = float(np.linalg.norm(span))
        count = max(int(math.ceil(length / float(step_px))), 1)
        samples.extend(start + span * (index / count) for index in range(count + 1))
    return samples


def radius_within_envelope(nominal_range_m: float) -> bool:
    """F2, applied once at registration time to the installed camera radius.

    The argument is the registered nominal distance from the camera to the
    subject's standing position, not a per-frame measurement, so no numerical
    tolerance is involved: a registered 3.50 m arc is inside the envelope by
    construction.  This becomes discriminating only when a future stage varies
    the radius.
    """
    low, high = DISTANCE_ENVELOPE_M
    return low <= float(nominal_range_m) <= high


def triangulation_angle_deg(eye_a, eye_b, joint) -> float:
    """Record-only: the angle the two rays subtend at the joint."""
    first = np.asarray(joint, dtype=float) - np.asarray(eye_a, dtype=float)
    second = np.asarray(joint, dtype=float) - np.asarray(eye_b, dtype=float)
    norms = float(np.linalg.norm(first)) * float(np.linalg.norm(second))
    if norms < 1.0e-12:
        raise ObjectiveError("Degenerate triangulation geometry.")
    return float(math.degrees(math.acos(float(np.clip(first @ second / norms, -1.0, 1.0)))))


def rank_candidates(evaluations: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Feasible candidates by descending score; ties break on candidate id.

    Infeasible candidates are dropped rather than ranked last: an infeasible
    viewpoint has no meaningful position in an ordering of usable ones.
    """
    feasible = [key for key, value in evaluations.items() if value.get("feasible")]
    return sorted(feasible, key=lambda key: (-float(evaluations[key]["score"]), key))


def select_best(evaluations: Mapping[str, Mapping[str, Any]]) -> str | None:
    ordered = rank_candidates(evaluations)
    return ordered[0] if ordered else None
