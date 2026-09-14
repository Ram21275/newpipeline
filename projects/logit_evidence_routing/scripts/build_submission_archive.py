#!/usr/bin/env python3
"""Build a deterministic, internally hashed result archive for paper audit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def collect(root: Path, includes: list[str], output: Path) -> list[Path]:
    files: list[Path] = []
    output_resolved = output.resolve()
    for name in includes:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"include path must stay below root: {name}")
        source = root / relative
        if not source.exists():
            raise RuntimeError(f"included result path is missing: {source}")
        candidates = [source] if source.is_file() else sorted(source.rglob("*"))
        for path in candidates:
            if path.is_symlink():
                raise RuntimeError(f"archive refuses symlink: {path}")
            if path.is_file() and path.resolve() != output_resolved:
                files.append(path)
    unique = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    if not unique:
        raise RuntimeError("archive input is empty")
    return unique


def add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = normalized(tarfile.TarInfo(name))
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def verify_internal_manifest(archive: Path, expected: dict[str, Any]) -> None:
    """Reopen the finished archive and verify every embedded byte stream."""

    with tarfile.open(archive, mode="r:gz") as tar:
        members = {member.name: member for member in tar.getmembers() if member.isfile()}
        if set(members) != {"ARCHIVE_MANIFEST.json", *(
                entry["path"] for entry in expected["files"])}:
            raise RuntimeError("archive members differ from the internal manifest")
        manifest_handle = tar.extractfile(members["ARCHIVE_MANIFEST.json"])
        if manifest_handle is None:
            raise RuntimeError("archive manifest cannot be read")
        embedded = json.load(manifest_handle)
        if embedded != expected:
            raise RuntimeError("embedded archive manifest differs from the build manifest")
        for entry in expected["files"]:
            handle = tar.extractfile(members[entry["path"]])
            if handle is None:
                raise RuntimeError(f"archive member cannot be read: {entry['path']}")
            digest = hashlib.sha256()
            byte_count = 0
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                byte_count += len(chunk)
            if byte_count != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                raise RuntimeError(f"archive member hash differs: {entry['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--include", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not root.is_dir() or root not in output.parents:
        raise RuntimeError("output must be located below the archive root")
    files = collect(root, args.include, output)
    entries: list[dict[str, Any]] = []
    for path in files:
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "schema_version": 1,
        "purpose": "deterministic_lger_submission_result_archive",
        "official_test_images_used": 0,
        "file_count": len(entries),
        "files": entries,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in files:
                    arcname = path.relative_to(root).as_posix()
                    info = normalized(tar.gettarinfo(str(path), arcname=arcname))
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
                add_bytes(tar, "ARCHIVE_MANIFEST.json", manifest_bytes)
    temporary.replace(output)
    verify_internal_manifest(output, manifest)
    digest = sha256(output)
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "archive": output.name,
        "archive_sha256": digest,
        "archive_bytes": output.stat().st_size,
        "internal_file_count": len(entries),
        "internal_manifest_verified": True,
        "deterministic_metadata": True,
        "official_test_images_used": 0,
    }
    audit_path = output.with_suffix(output.suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
