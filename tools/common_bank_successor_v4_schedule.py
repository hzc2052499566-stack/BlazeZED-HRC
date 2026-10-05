"""Result-blind, SHA-seeded capture schedule for the prospective v4 probe."""

from __future__ import annotations

import hashlib

CHARACTERS = ("F01", "F02", "M01", "M02")
REPEATS = (1, 2, 3)
BRIDGE_ID = "v4qa_az000_el00"
SEED = "successor-v4-full-prior-sector-probe-v1"


def view_id(azimuth: float, elevation: float) -> str:
    def enc(value: float) -> str:
        sign = "m" if value < 0 else "p"
        return sign + f"{abs(value):04.1f}".replace(".", "")
    return f"v4a{int(azimuth * 10):04d}e{enc(elevation)}"


AZIMUTHS = (7.5, 22.5, 37.5, 52.5, 67.5, 82.5, 97.5, 112.5)
ELEVATIONS = (-2.5, 5.0, 12.5)
CANDIDATE_IDS = tuple(view_id(a, e) for a in AZIMUTHS for e in ELEVATIONS)


def session_schedule(character: str, repeat: int, seed: str = SEED) -> tuple[dict, ...]:
    if character not in CHARACTERS or repeat not in REPEATS:
        raise ValueError("Unknown character or repeat.")
    ordered = sorted(CANDIDATE_IDS, key=lambda item: hashlib.sha256(f"{seed}|{character}|{item}".encode()).hexdigest())
    rows = []
    rotation = repeat - 1
    for batch_index in range(8):
        triple = ordered[batch_index * 3:(batch_index + 1) * 3]
        primaries = triple[rotation:] + triple[:rotation]
        rows.append({"batch_index": batch_index + 1, "view_ids": [BRIDGE_ID, *primaries]})
    return tuple(rows)


def full_schedule(seed: str = SEED) -> tuple[dict, ...]:
    sessions = []
    identities = [(c, r) for c in CHARACTERS for r in REPEATS]
    identities.sort(key=lambda x: hashlib.sha256(f"{seed}|session|{x[0]}|{x[1]}".encode()).hexdigest())
    for index, (character, repeat) in enumerate(identities, 1):
        sessions.append({"execution_index": index, "session_id": f"v4p1_{character}_r{repeat}",
                         "character": character, "repeat": repeat,
                         "batches": list(session_schedule(character, repeat, seed))})
    return tuple(sessions)
