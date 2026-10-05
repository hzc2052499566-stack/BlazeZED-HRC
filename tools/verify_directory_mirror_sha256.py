"""Verify a directory mirror with per-file SHA-256 hashes.

The verifier is intentionally read-only for the source and mirrored directory. It
writes a canonical CSV manifest and a compact JSON report outside those trees.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


CHUNK_BYTES = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, tuple[Path, int]]:
    entries: dict[str, tuple[Path, int]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in entries:
            raise RuntimeError(f"Duplicate relative path: {relative}")
        entries[relative] = (path, path.stat().st_size)
    return entries


def atomic_write_csv(path: Path, rows: list[tuple[str, int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("relative_path", "bytes", "sha256"))
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve(strict=True)
    destination = args.destination.resolve(strict=True)
    if not source.is_dir() or not destination.is_dir():
        raise RuntimeError("Source and destination must both be directories.")
    if source == destination:
        raise RuntimeError("Source and destination resolve to the same directory.")

    source_entries = inventory(source)
    destination_entries = inventory(destination)
    source_names = set(source_entries)
    destination_names = set(destination_entries)
    missing = sorted(source_names - destination_names)
    extra = sorted(destination_names - source_names)
    mismatches: list[dict[str, object]] = []
    rows: list[tuple[str, int, str]] = []

    names = sorted(source_names & destination_names)
    for index, relative in enumerate(names, start=1):
        source_path, source_size = source_entries[relative]
        destination_path, destination_size = destination_entries[relative]
        if source_size != destination_size:
            mismatches.append(
                {
                    "relative_path": relative,
                    "reason": "size",
                    "source_bytes": source_size,
                    "destination_bytes": destination_size,
                }
            )
            continue
        source_hash = sha256_file(source_path)
        destination_hash = sha256_file(destination_path)
        if source_hash != destination_hash:
            mismatches.append(
                {
                    "relative_path": relative,
                    "reason": "sha256",
                    "source_sha256": source_hash,
                    "destination_sha256": destination_hash,
                }
            )
            continue
        rows.append((relative, source_size, source_hash))
        if args.progress_every > 0 and index % args.progress_every == 0:
            print(f"[verify] {index}/{len(names)} files", flush=True)

    passed = not missing and not extra and not mismatches and len(rows) == len(source_entries)
    report: dict[str, object] = {
        "schema": "directory_mirror_sha256_verification_v1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "destination": str(destination),
        "source_file_count": len(source_entries),
        "destination_file_count": len(destination_entries),
        "source_bytes": sum(size for _, size in source_entries.values()),
        "destination_bytes": sum(size for _, size in destination_entries.values()),
        "verified_file_count": len(rows),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "mismatch_count": len(mismatches),
        "missing_examples": missing[:20],
        "extra_examples": extra[:20],
        "mismatch_examples": mismatches[:20],
        "status": "PASS" if passed else "FAIL",
    }

    if passed:
        atomic_write_csv(args.manifest, rows)
        report["manifest"] = str(args.manifest.resolve())
        report["manifest_sha256"] = sha256_file(args.manifest.resolve())
    atomic_write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # pragma: no cover - command-line safety boundary
        print(f"VERIFY_DIRECTORY_MIRROR_FAILED: {error}", file=sys.stderr)
        raise
