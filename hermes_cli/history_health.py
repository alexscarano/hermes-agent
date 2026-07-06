import json, os
from pathlib import Path

def health_check():
    from hermes_cli.config import cfg_get
    from hermes_cli.history import history_dir, list_snapshots

    hd = history_dir()
    enabled = bool(cfg_get('history.enabled', default=True))
    exists = hd.is_dir()

    if exists:
        snaps = list_snapshots(limit=9999)
        total_bytes = sum(
            f.stat().st_size
            for f in hd.rglob('*')
            if f.is_file()
        )
    else:
        snaps = []
        total_bytes = 0

    return {
        'enabled': enabled,
        'exists': exists,
        'snapshot_count': len(snaps),
        'total_size_mb': round(total_bytes / (1024 * 1024), 2),
    }
