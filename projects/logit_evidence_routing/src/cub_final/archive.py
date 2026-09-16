"""Deterministic portable archive with an internal checksum manifest."""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path
from typing import Iterable

from .core import atomic_write_json, file_sha256


def _normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def build_archive(root: Path, includes: Iterable[Path], output: Path) -> dict[str, object]:
    root = root.resolve()
    output = output.resolve()
    if root not in output.parents:
        raise ValueError("archive output must be located below root")
    files: list[Path] = []
    for relative in includes:
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe include path: {relative}")
        source = root / relative
        if not source.exists():
            raise FileNotFoundError(source)
        values = [source] if source.is_file() else sorted(source.rglob("*"))
        for value in values:
            if value.is_symlink():
                raise ValueError(f"archive refuses symlink: {value}")
            if (
                "__pycache__" in value.parts
                or any(part.endswith(".egg-info") for part in value.parts)
                or value.suffix in {".pyc", ".pyo"}
            ):
                continue
            if value.is_file() and value.resolve() != output:
                files.append(value)
    files = sorted(set(files), key=lambda value: value.relative_to(root).as_posix())
    manifest = {
        "schema_version": 1,
        "files": [
            {
                "path": value.relative_to(root).as_posix(),
                "bytes": value.stat().st_size,
                "sha256": file_sha256(value),
            }
            for value in files
        ],
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for value in files:
                    name = value.relative_to(root).as_posix()
                    info = _normalized(tar.gettarinfo(str(value), arcname=name))
                    with value.open("rb") as handle:
                        tar.addfile(info, handle)
                info = _normalized(tarfile.TarInfo("ARCHIVE_MANIFEST.json"))
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    temporary.replace(output)
    audit: dict[str, object] = {
        "status": "PASS",
        "archive": str(output),
        "archive_sha256": file_sha256(output),
        "archive_bytes": output.stat().st_size,
        "file_count": len(files),
    }
    atomic_write_json(Path(str(output) + ".audit.json"), audit)
    Path(str(output) + ".sha256").write_text(
        f"{audit['archive_sha256']}  {output.name}\n", encoding="ascii"
    )
    return audit
