"""Deterministic provenance manifests for dataset source files.

This module defines the interchange contract between an official-product
converter and the existing NumPy-based dataset loaders. It deliberately does
not guess the internal schema of TACO, SEN2Vénus archives, or Bhoonidhi
products. A converter for one of those products should first materialize its
validated LR/HR arrays, then call :func:`build_pair_manifest` before training.

A manifest is sealed with a SHA-256 checksum over its canonical JSON body and
contains SHA-256 checksums for every declared source file. Loading a manifest
can therefore fail closed when a file is changed, removed, or the manifest is
edited after creation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


MANIFEST_VERSION = 1


class ProvenanceError(ValueError):
    """Raised when a provenance manifest is malformed or has been tampered with."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally so large imagery is never loaded in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(root: Path, path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        return candidate.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ProvenanceError(f"source file must be inside manifest root: {path}") from exc


def _seal(manifest: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    body["manifest_sha256"] = _sha256_bytes(_canonical_bytes(body))
    return body


def _verify_seal(manifest: Mapping[str, Any]) -> None:
    recorded = manifest.get("manifest_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ProvenanceError("manifest_sha256 is missing or malformed")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    expected = _sha256_bytes(_canonical_bytes(body))
    if recorded != expected:
        raise ProvenanceError(
            f"manifest seal mismatch: recorded {recorded}, expected {expected}"
        )


def build_pair_manifest(
    root: str | Path,
    product: str,
    sensor: str,
    pairs: Iterable[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a sealed manifest for declared LR/HR NumPy pair files.

    Each pair must contain ``id``, ``lr_path``, and ``hr_path``. Paths may be
    absolute or relative to ``root`` but must resolve inside ``root``. Extra
    pair metadata is retained under ``metadata`` and is not silently inferred.
    """
    root_path = Path(root).resolve()
    normalized_pairs: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_files: set[str] = set()

    for pair in pairs:
        pair_id = pair.get("id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ProvenanceError("each pair needs a non-empty string id")
        if pair_id in seen_ids:
            raise ProvenanceError(f"duplicate pair id: {pair_id}")
        seen_ids.add(pair_id)
        pair_files: dict[str, str] = {}
        for role in ("lr_path", "hr_path", "lr_mask_path", "hr_mask_path"):
            if role not in pair:
                if role.endswith("_mask_path"):
                    continue
                raise ProvenanceError(f"pair {pair_id!r} is missing {role}")
            rel = _relative_path(root_path, pair[role])
            absolute = root_path / rel
            if not absolute.is_file():
                raise ProvenanceError(f"pair {pair_id!r} file does not exist: {absolute}")
            pair_files[role] = rel
            if rel not in seen_files:
                files.append(
                    {
                        "path": rel,
                        "size_bytes": absolute.stat().st_size,
                        "sha256": sha256_file(absolute),
                    }
                )
                seen_files.add(rel)
        entry = {"id": pair_id, **pair_files}
        extra = pair.get("metadata")
        if extra is not None:
            if not isinstance(extra, Mapping):
                raise ProvenanceError(f"pair {pair_id!r} metadata must be an object")
            entry["metadata"] = dict(extra)
        normalized_pairs.append(entry)

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "product": product,
        "sensor": sensor,
        "root": ".",
        "files": files,
        "pairs": normalized_pairs,
        "metadata": dict(metadata or {}),
    }
    return _seal(manifest)


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Atomically write a sealed manifest as canonical, diff-friendly JSON."""
    sealed = _seal(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(sealed, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_manifest(
    path: str | Path,
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Read, validate, and optionally checksum all files named by a manifest."""
    manifest_path = Path(path).resolve()
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"failed to read manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ProvenanceError("manifest root must be a JSON object")
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ProvenanceError(f"unsupported manifest_version: {manifest.get('manifest_version')!r}")
    for key in ("product", "sensor", "files", "pairs"):
        if key not in manifest:
            raise ProvenanceError(f"manifest is missing required field: {key}")
    _verify_seal(manifest)

    if verify_files:
        root = manifest_path.parent
        expected = {item.get("path"): item for item in manifest["files"]}
        for pair in manifest["pairs"]:
            if not isinstance(pair, dict) or not isinstance(pair.get("id"), str):
                raise ProvenanceError("each manifest pair must be an object with a string id")
            for role in ("lr_path", "hr_path", "lr_mask_path", "hr_mask_path"):
                rel = pair.get(role)
                # BUG FIX (2026-09-21, agent4, T25 turn): mask-path roles are
                # optional by design (build_pair_manifest's write side already
                # treats them this way — it skips them entirely when absent).
                # This read-side check did NOT make the same exception, so
                # `rel is None` for an intentionally-absent mask path was
                # rejected as "undeclared or unsafe" — meaning ANY manifest
                # without masks (the common case, and the only case any
                # existing test or caller actually produced) was completely
                # unreadable via read_manifest(..., verify_files=True).
                # Confirmed by direct reproduction before fixing, not assumed.
                if role.endswith("_mask_path") and rel is None:
                    continue
                if (
                    not isinstance(rel, str)
                    or Path(rel).is_absolute()
                    or ".." in Path(rel).parts
                    or rel not in expected
                ):
                    raise ProvenanceError(
                        f"pair {pair['id']!r} references undeclared or unsafe {role}: {rel!r}"
                    )
        for rel, item in expected.items():
            if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise ProvenanceError(f"invalid manifest-relative file path: {rel!r}")
            source = (root / rel).resolve()
            if not source.is_file():
                raise ProvenanceError(f"manifest source file is missing: {source}")
            actual_size = source.stat().st_size
            actual_hash = sha256_file(source)
            if actual_size != item.get("size_bytes") or actual_hash != item.get("sha256"):
                raise ProvenanceError(f"source checksum mismatch: {rel}")
    return manifest


def load_numpy_pair(manifest_path: str | Path, pair_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one verified pair from a manifest for an existing Dataset adapter."""
    manifest = read_manifest(manifest_path, verify_files=True)
    pair = next((item for item in manifest["pairs"] if item.get("id") == pair_id), None)
    if pair is None:
        raise ProvenanceError(f"pair id not found in manifest: {pair_id}")
    root = Path(manifest_path).resolve().parent
    lr = np.load(root / pair["lr_path"])
    hr = np.load(root / pair["hr_path"])
    return lr, hr, dict(pair.get("metadata") or {})


if __name__ == "__main__":
    print("provenance manifest helpers ready")
