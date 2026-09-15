#!/usr/bin/env python3
"""Build a deterministic small-file-only package for a fresh Kaggle account."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any


BANNED_SUFFIXES = {
    ".bin", ".ckpt", ".npy", ".npz", ".pt", ".pth", ".safetensors",
}
BANNED_PARTS = {"cache", "caches", "checkpoint", "checkpoints", "records", "weights"}


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
    info.mode = 0o644
    return info


def parse_file(value: str, *, maximum_bytes: int) -> tuple[Path, str]:
    if "::" not in value:
        raise RuntimeError("--file values must use SOURCE::ARCHIVE_NAME")
    source_text, archive_name = value.split("::", 1)
    source = Path(source_text).resolve()
    relative = Path(archive_name)
    if not source.is_file():
        raise RuntimeError(f"transfer source is missing: {source}")
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RuntimeError(f"unsafe archive name: {archive_name}")
    lower_parts = {part.casefold() for part in relative.parts}
    if source.suffix.casefold() in BANNED_SUFFIXES or lower_parts & BANNED_PARTS:
        raise RuntimeError(f"large-cache/model artifact is forbidden: {archive_name}")
    if source.stat().st_size > maximum_bytes:
        raise RuntimeError(
            f"transfer source exceeds {maximum_bytes} bytes: {source}"
        )
    return source, relative.as_posix()


def add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = normalized(tarfile.TarInfo(name))
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def verify(archive: Path, manifest: dict[str, Any]) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        members = {member.name: member for member in tar.getmembers() if member.isfile()}
        expected = {"TRANSFER_MANIFEST.json", *(row["path"] for row in manifest["files"])}
        if set(members) != expected:
            raise RuntimeError("transfer archive members differ from its manifest")
        handle = tar.extractfile(members["TRANSFER_MANIFEST.json"])
        if handle is None or json.load(handle) != manifest:
            raise RuntimeError("embedded transfer manifest differs")
        for row in manifest["files"]:
            handle = tar.extractfile(members[row["path"]])
            if handle is None:
                raise RuntimeError(f"cannot read transfer member: {row['path']}")
            payload = handle.read()
            if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise RuntimeError(f"transfer member hash differs: {row['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file", action="append", required=True, metavar="SOURCE::ARCHIVE_NAME"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-file-bytes", type=int, default=5_000_000)
    args = parser.parse_args()
    if args.maximum_file_bytes <= 0:
        raise RuntimeError("maximum file size must be positive")
    files = [
        parse_file(value, maximum_bytes=args.maximum_file_bytes) for value in args.file
    ]
    names = [name for _, name in files]
    if len(names) != len(set(names)):
        raise RuntimeError("transfer archive names must be unique")
    files.sort(key=lambda item: item[1])
    entries = [
        {"path": name, "bytes": source.stat().st_size, "sha256": sha256(source)}
        for source, name in files
    ]
    manifest = {
        "schema_version": 1,
        "purpose": "fresh_kaggle_small_file_transfer",
        "large_model_weights_included": False,
        "large_activation_caches_included": False,
        "official_test_images_included": False,
        "official_test_images_used": 0,
        "maximum_file_bytes": args.maximum_file_bytes,
        "file_count": len(entries),
        "files": entries,
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for source, name in files:
                    info = normalized(tar.gettarinfo(str(source), arcname=name))
                    with source.open("rb") as handle:
                        tar.addfile(info, handle)
                add_bytes(tar, "TRANSFER_MANIFEST.json", payload)
    temporary.replace(args.output)
    verify(args.output, manifest)
    checksum = sha256(args.output)
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    sidecar.write_text(f"{checksum}  {args.output.name}\n", encoding="ascii")
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "archive": str(args.output.resolve()),
        "archive_bytes": args.output.stat().st_size,
        "archive_sha256": checksum,
        "file_count": len(entries),
        "internal_manifest_verified": True,
        "large_artifacts_included": False,
        "official_test_images_used": 0,
    }
    audit_path = args.output.with_suffix(args.output.suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
