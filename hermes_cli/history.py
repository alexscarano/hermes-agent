#!/usr/bin/env python3
"""Time Travel — .hermes-history snapshot system.

Automatic file versioning that silently captures file contents before and after
any Hermes file modification (patch, write_file, skill_manage). Enables
immediate rollback if self-improvement loops go wrong.

Architecture:
  .hermes-history/
    <YYYYMMDD-HHMMSS-ffffff>/
      meta.json       {id, timestamp, files: {name: {path, operation, sha_before, sha_after}}}
      before/          original file(s)
      after/           modified file(s)
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_SNAPSHOTS = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hermes_home() -> Path:
    """Return the absolute Hermes home directory."""
    from hermes_constants import get_hermes_home
    return Path(get_hermes_home())


def _history_enabled() -> bool:
    """Check whether the history system is enabled in config."""
    try:
        from hermes_cli.config import cfg_get
        return bool(cfg_get("history.enabled", default=True))
    except Exception:
        return True


def _snapshot_id() -> str:
    """Return a unique snapshot id (sortable by name)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _sha256(path: str | Path) -> str:
    """Return the first 12 hex chars of the file's SHA-256."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def history_dir() -> Path:
    """Return the absolute path to the .hermes-history directory."""
    return _hermes_home() / ".hermes-history"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_filename(path: str | Path) -> str:
    """Return a filesystem-safe name for the given path."""
    return str(Path(path).resolve()).replace("/", "_")


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def snapshot_before(file_path: str, operation: str = "unknown") -> str | None:
    """Copy the original file to a before-snapshot.

    Args:
        file_path: Absolute or relative path to the file.
        operation: Label describing what triggered the snapshot (e.g. "patch", "write_file").

    Returns:
        Snapshot ID string, or None if history is disabled, the file doesn't
        exist, or any error occurs. Never raises.
    """
    try:
        if not _history_enabled():
            return None

        src = Path(file_path).resolve()
        if not src.is_file():
            return None

        sid = _snapshot_id()
        safe = _safe_filename(src)
        snap_dir = _ensure_dir(history_dir() / sid)

        # Copy original into before/
        before_dir = _ensure_dir(snap_dir / "before")
        shutil.copy2(str(src), str(before_dir / safe))

        # Write meta.json
        meta = {
            "id": sid,
            "timestamp": datetime.now().isoformat(),
            "files": {
                safe: {
                    "path": str(src),
                    "operation": operation,
                    "sha_before": _sha256(src),
                    "sha_after": None,
                }
            },
        }
        (snap_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )

        _prune_old()
        return sid

    except Exception:
        logger.debug("snapshot_before failed for %s", file_path, exc_info=True)
        return None


def snapshot_after(
    file_path: str, snapshot_id: str | None, operation: str = "unknown"
) -> str | None:
    """Copy the modified file into the after-snapshot and update meta.

    Args:
        file_path: Absolute or relative path to the modified file.
        snapshot_id: The ID returned by the corresponding snapshot_before call.
        operation: Label describing the operation.

    Returns:
        Snapshot ID, or None if no snapshot_id is given, the file doesn't
        exist, or any error occurs. Never raises.
    """
    if not snapshot_id:
        return None

    try:
        src = Path(file_path).resolve()
        if not src.is_file():
            return None

        safe = _safe_filename(src)
        snap_dir = history_dir() / snapshot_id
        after_dir = _ensure_dir(snap_dir / "after")
        shutil.copy2(str(src), str(after_dir / safe))

        # Update meta.json
        meta_path = snap_dir / "meta.json"
        if meta_path.exists():
            meta: dict[str, Any] = json.loads(meta_path.read_text())
        else:
            meta = {
                "id": snapshot_id,
                "timestamp": datetime.now().isoformat(),
                "files": {},
            }

        meta.setdefault("files", {}).setdefault(safe, {})
        meta["files"][safe]["path"] = str(src)
        meta["files"][safe]["operation"] = operation
        meta["files"][safe]["sha_after"] = _sha256(src)

        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return snapshot_id

    except Exception:
        logger.debug("snapshot_after failed for %s", file_path, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def _prune_old() -> None:
    """Remove snapshots beyond MAX_SNAPSHOTS. Called automatically after each
    ``snapshot_before``."""
    hd = history_dir()
    if not hd.is_dir():
        return
    snapshots = sorted(hd.iterdir(), reverse=True)
    for old in snapshots[MAX_SNAPSHOTS:]:
        shutil.rmtree(str(old), ignore_errors=True)


def prune(keep: int = 50) -> int:
    """Manually purge all but the *keep* most recent snapshots.

    Returns the number of snapshots removed.
    """
    hd = history_dir()
    if not hd.is_dir():
        return 0
    snapshots = sorted(hd.iterdir(), reverse=True)
    removed = 0
    for old in snapshots[keep:]:
        shutil.rmtree(str(old), ignore_errors=True)
        removed += 1
    return removed


# ---------------------------------------------------------------------------
# Inspection & Rollback
# ---------------------------------------------------------------------------

def list_snapshots(limit: int = 20) -> list[dict[str, Any]]:
    """Return recent snapshots sorted newest-first.

    Each entry is the content of the snapshot's ``meta.json``.
    """
    hd = history_dir()
    if not hd.is_dir():
        return []
    snapshots: list[dict[str, Any]] = []
    for entry in sorted(hd.iterdir(), reverse=True)[:limit]:
        meta_file = entry / "meta.json"
        if meta_file.exists():
            try:
                snapshots.append(json.loads(meta_file.read_text()))
            except Exception:
                pass
    return snapshots


def show_diff(snapshot_id: str) -> str:
    """Return a unified diff of all files in the snapshot."""
    snap_dir = history_dir() / snapshot_id
    meta_file = snap_dir / "meta.json"
    if not meta_file.exists():
        return f"Snapshot {snapshot_id} not found"

    meta: dict[str, Any] = json.loads(meta_file.read_text())
    parts: list[str] = []

    for safe, info in meta.get("files", {}).items():
        before_f = snap_dir / "before" / safe
        after_f = snap_dir / "after" / safe

        before_text = before_f.read_text() if before_f.exists() else ""
        after_text = after_f.read_text() if after_f.exists() else ""

        diff = difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=f"before/{info.get('path', safe)}",
            tofile=f"after/{info.get('path', safe)}",
        )
        parts.append("".join(diff))

    return "\n".join(parts)


def rollback(snapshot_id: str) -> list[str]:
    """Restore files from a snapshot's ``before/`` directory.

    Returns a list of restored file paths.
    """
    restored: list[str] = []
    snap_dir = history_dir() / snapshot_id
    before_dir = snap_dir / "before"
    meta_file = snap_dir / "meta.json"

    if not meta_file.exists():
        return restored

    meta: dict[str, Any] = json.loads(meta_file.read_text())

    for safe, info in meta.get("files", {}).items():
        src = before_dir / safe
        dst_str = info.get("path", "")
        if not dst_str or not src.exists():
            continue
        try:
            dst = Path(dst_str)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            restored.append(str(dst))
        except Exception:
            logger.debug("rollback failed for %s -> %s", src, dst_str, exc_info=True)

    return restored