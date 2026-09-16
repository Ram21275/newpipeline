"""Crash-safe persistence, hashing, and environment inventory."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(canonical_json(row) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            output.append(value)
    return output


class ShardWriter:
    """One atomic JSON record per stable work key, safe across restarts."""

    def __init__(self, root: Path, *, config_hash: str) -> None:
        self.root = root
        self.config_hash = config_hash
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key_name(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json"

    def completed(self, key: str) -> bool:
        path = self.root / self._key_name(key)
        if not path.is_file():
            return False
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            value.get("key") == key
            and value.get("config_hash") == self.config_hash
            and value.get("status") in {"complete", "excluded", "failed"}
        )

    def read(self, key: str) -> dict[str, Any] | None:
        """Return a same-config terminal shard, or ``None`` if it is unusable."""

        path = self.root / self._key_name(key)
        if not self.completed(key):
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value

    def write(self, key: str, payload: dict[str, Any], *, status: str = "complete") -> Path:
        if status not in {"complete", "excluded", "failed"}:
            raise ValueError("invalid terminal shard status")
        path = self.root / self._key_name(key)
        atomic_write_json(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "key": key,
                "config_hash": self.config_hash,
                "status": status,
                "payload": payload,
            },
        )
        return path

    def consolidate(self, destination: Path) -> int:
        records: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("config_hash") != self.config_hash:
                continue
            records.append(value)
        records.sort(key=lambda row: str(row["key"]))
        atomic_write_jsonl(destination, records)
        return len(records)


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_inventory() -> dict[str, Any]:
    inventory: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }
    try:
        import torch

        inventory["torch"] = torch.__version__
        inventory["cuda_version"] = torch.version.cuda
        inventory["cuda_available"] = torch.cuda.is_available()
        inventory["cuda_device_count"] = torch.cuda.device_count()
        inventory["bf16_supported"] = bool(
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        )
        inventory["gpus"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    except ImportError:
        inventory["torch"] = None
        inventory["cuda_available"] = False
    for package in ("transformers", "accelerate", "bitsandbytes", "safetensors"):
        try:
            module = __import__(package)
            inventory[package] = getattr(module, "__version__", "unknown")
        except (ImportError, OSError):
            inventory[package] = None
    return inventory
