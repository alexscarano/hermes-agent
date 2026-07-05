"""CLI command for managing .hermes-history snapshots."""

from __future__ import annotations

import json
import textwrap


def cmd_history(args) -> None:
    """Manage .hermes-history snapshots (time-travel)."""
    sub = getattr(args, "history_command", None) or "list"

    from hermes_cli.history import list_snapshots, show_diff, rollback, prune

    if sub == "list":
        snaps = list_snapshots(limit=getattr(args, "limit", 20))
        if not snaps:
            print("No snapshots in .hermes-history/")
            return
        print(f"Snapshots ({len(snaps)}):")
        print()
        for i, snap in enumerate(snaps, 1):
            sid = snap.get("id", "?")
            ts = snap.get("timestamp", "?")
            files = snap.get("files", {})
            ops = set()
            srcs = []
            for info in files.values():
                ops.add(info.get("operation", "?"))
                srcs.append(info.get("path", "?"))
            label = ", ".join(sorted(ops))
            print(f"  {i:>3}. {sid}")
            print(f"       {ts}")
            print(f"       {label}: {', '.join(srcs)}")
            print()

    elif sub == "diff":
        sid = getattr(args, "snapshot_id", None)
        if not sid:
            print("Usage: hermes history diff <snapshot-id>")
            return
        out = show_diff(sid)
        if out:
            print(out)
        else:
            print(f"No diff for snapshot {sid}")

    elif sub == "rollback":
        sid = getattr(args, "snapshot_id", None)
        if not sid:
            print("Usage: hermes history rollback <snapshot-id>")
            return
        restored = rollback(sid)
        if restored:
            print(f"Restored {len(restored)} file(s):")
            for f in restored:
                print(f"  {f}")
        else:
            print(f"No files restored from {sid}")

    elif sub == "prune":
        keep = getattr(args, "keep", 50)
        removed = prune(keep=keep)
        print(f"Removed {removed} snapshot(s), keeping {keep}")

    elif sub == "cat":
        """Show full meta.json content of a snapshot."""
        sid = getattr(args, "snapshot_id", None)
        if not sid:
            print("Usage: hermes history cat <snapshot-id>")
            return
        from hermes_cli.history import history_dir

        meta_file = history_dir() / sid / "meta.json"
        if meta_file.exists():
            print(meta_file.read_text())
        else:
            print(f"Snapshot {sid} not found")

    else:
        print(f"Unknown history subcommand: {sub}")