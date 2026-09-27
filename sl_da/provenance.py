"""What exactly produced a file: model revisions, library versions, machine, time.

A path or a repo name proves nothing a week later. /workspace is wiped with the pod, the
same corpus name gets reused, and an HF repo name points at whatever was pushed last. So
every artifact that a student is trained on, and every student, records the COMMIT of each
model it touched and the versions of the code that touched it.
"""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import time
from pathlib import Path


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sha256_json(obj) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def model_revision(name_or_path: str | None) -> dict | None:
    """The commit a HF repo resolved to, or a content hash for a local directory.

    Uses the local cache when it can (the model has already been downloaded by whatever
    loaded it), so this costs nothing and works offline. Never raises: a provenance
    helper that can crash a 4-hour run is worse than one that records an error.
    """
    if not name_or_path:
        return None
    p = Path(name_or_path)
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.is_file())
        h = hashlib.sha256()
        for f in files:
            h.update(str(f.relative_to(p)).encode())
            h.update(sha256_file(f).encode())
        return {"source": "local_dir", "path": str(p.resolve()), "n_files": len(files),
                "content_sha256": h.hexdigest()}
    try:
        from huggingface_hub import snapshot_download
        snap = Path(snapshot_download(name_or_path, allow_patterns=["config.json",
                                                                    "adapter_config.json"]))
        return {"source": "hf", "repo": name_or_path, "commit": snap.name}
    except Exception as e:                      # noqa: BLE001 -- see docstring
        return {"source": "hf", "repo": name_or_path, "commit": None, "error": repr(e)}


def environment() -> dict:
    """Library versions, GPU, host and wall-clock time."""
    import importlib.metadata as md
    env = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": socket.gethostname(), "python": platform.python_version()}
    for lib in ("torch", "transformers", "peft", "vllm", "numpy"):
        try:
            env[lib] = md.version(lib)
        except md.PackageNotFoundError:
            env[lib] = None
    try:
        import torch
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:                           # noqa: BLE001
        env["gpu"] = None
    return env
