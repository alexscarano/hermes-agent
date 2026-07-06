"""System Sensor Checks — Reactive cron triggers.

Each sensor is a registered check function that evaluates a system condition
and returns True when the condition is met (trigger fire) or False otherwise.

Sensors are called from inside the cron tick loop, so they must be fast
(sub-millisecond to low-millisecond). Expensive checks (ping, procfs reads)
are internally cached and rate-limited.

Usage:
    from cron.sensors import check_sensor, list_sensors

    # Check if CPU > 80%
    triggered = check_sensor("cpu", {"threshold": 80, "operator": ">"})
    
    # List all registered sensor types
    print(list_sensors())
"""

from __future__ import annotations

import logging
import os
import re
import stat
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sensor registry
# ---------------------------------------------------------------------------

SensorFn = Callable[..., bool]
_sensors: dict[str, SensorFn] = {}
_sensors_lock = threading.Lock()


def register_sensor(name: str) -> Callable[[SensorFn], SensorFn]:
    """Decorator to register a sensor check function.

    The function receives **kwargs matching the params dict from the
    schedule definition.
    """
    def decorator(fn: SensorFn) -> SensorFn:
        with _sensors_lock:
            _sensors[name] = fn
        return fn
    return decorator


def list_sensors() -> list[str]:
    """Return sorted list of registered sensor names."""
    with _sensors_lock:
        return sorted(_sensors.keys())


def check_sensor(sensor_type: str, params: dict[str, Any] | None = None) -> bool:
    """Evaluate a sensor.

    Args:
        sensor_type: Registered sensor name (e.g. "cpu", "memory").
        params: Parameters dict passed to the check function.

    Returns:
        True when the condition is triggered (job should fire).

    Raises:
        ValueError: If sensor_type is not registered.
    """
    with _sensors_lock:
        fn = _sensors.get(sensor_type)
    if fn is None:
        raise ValueError(
            f"Unknown sensor '{sensor_type}'. "
            f"Registered: {', '.join(sorted(_sensors))}"
        )
    try:
        return bool(fn(**(params or {})))
    except Exception as e:
        logger.warning("Sensor '%s' check failed: %s", sensor_type, e, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Math helper
# ---------------------------------------------------------------------------

def _compare(value: float, threshold: float, operator: str = ">") -> bool:
    """Compare value against threshold using operator."""
    if operator == ">":
        return value > threshold
    elif operator == ">=":
        return value >= threshold
    elif operator == "<":
        return value < threshold
    elif operator == "<=":
        return value <= threshold
    elif operator == "==":
        return abs(value - threshold) < 0.001
    elif operator == "!=":
        return abs(value - threshold) >= 0.001
    logger.warning("Unknown sensor operator '%s', defaulting to >", operator)
    return value > threshold


# ---------------------------------------------------------------------------
# Procfs helpers (/proc/stat, /proc/meminfo)
# ---------------------------------------------------------------------------

# Cache for /proc/stat reads — avoid parsing on every call
_cpu_cache: dict[str, Any] = {}
_cpu_cache_lock = threading.Lock()
_CPU_CACHE_TTL = 1.0  # seconds


def _read_cpu_times() -> Optional[dict[str, float]]:
    """Parse /proc/stat for CPU times. Returns {user, nice, system, idle, iowait, ...}."""
    try:
        with open("/proc/stat", "r") as f:
            for line in f:
                if line.startswith("cpu "):
                    parts = line.split()
                    # cpu  user nice system idle iowait irq softirq steal guest guest_nice
                    fields = ["user", "nice", "system", "idle", "iowait",
                              "irq", "softirq", "steal"]
                    vals = {}
                    for i, name in enumerate(fields):
                        idx = i + 1
                        if idx < len(parts):
                            vals[name] = float(parts[idx])
                    return vals
    except (OSError, IOError, IndexError, ValueError) as e:
        logger.debug("Failed to read /proc/stat: %s", e)
    return None


def _calc_cpu_usage() -> Optional[float]:
    """Calculate CPU usage percentage over ~1s sample.

    Returns 0-100 float, or None on failure.
    """
    t1 = _read_cpu_times()
    if t1 is None:
        return None
    time.sleep(0.8)  # brief sample window
    t2 = _read_cpu_times()
    if t2 is None:
        return None
    idle1 = t1.get("idle", 0) + t1.get("iowait", 0)
    idle2 = t2.get("idle", 0) + t2.get("iowait", 0)
    total1 = sum(t1.values())
    total2 = sum(t2.values())
    delta_idle = idle2 - idle1
    delta_total = total2 - total1
    if delta_total <= 0:
        return 0.0
    return 100.0 * (1.0 - delta_idle / delta_total)


_single_cpu_usage: Optional[float] = None
_single_cpu_time: float = 0
_single_cpu_lock = threading.Lock()


def _get_single_cpu_usage() -> Optional[float]:
    """Thread-safe cached CPU usage with TTL."""
    global _single_cpu_usage, _single_cpu_time
    now = time.monotonic()
    with _single_cpu_lock:
        if _single_cpu_usage is not None and (now - _single_cpu_time) < _CPU_CACHE_TTL:
            return _single_cpu_usage
        val = _calc_cpu_usage()
        if val is not None:
            _single_cpu_usage = val
            _single_cpu_time = now
        return val


# ---------------------------------------------------------------------------
# Memory helper
# ---------------------------------------------------------------------------


def _read_meminfo() -> Optional[dict[str, int]]:
    """Parse /proc/meminfo into {MemTotal, MemAvailable, ...} in kB."""
    result: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val_str = parts[1].strip().lower()
                    # Extract numeric value (strip units like " kB", " mB")
                    val_match = re.match(r"(\d+)", val_str)
                    if val_match:
                        result[key] = int(val_match.group(1))
    except (OSError, IOError) as e:
        logger.debug("Failed to read /proc/meminfo: %s", e)
    return result or None


# ---------------------------------------------------------------------------
# Individual sensors
# ---------------------------------------------------------------------------


@register_sensor("cpu")
def check_cpu(threshold: float = 80, operator: str = ">") -> bool:
    """Check if CPU usage exceeds threshold.

    Args:
        threshold: Percentage threshold (0-100).
        operator: Comparison operator (>, >=, <, <=, ==, !=).

    Returns:
        True when condition is met.
    """
    usage = _get_single_cpu_usage()
    if usage is None:
        logger.debug("CPU sensor: cannot read /proc/stat")
        return False
    triggered = _compare(usage, threshold, operator)
    if triggered:
        logger.debug("CPU sensor triggered: %.1f%% %s %.0f%%", usage, operator, threshold)
    return triggered


@register_sensor("memory")
def check_memory(threshold: float = 90, operator: str = ">") -> bool:
    """Check if memory usage exceeds threshold.

    Reads MemTotal and MemAvailable from /proc/meminfo.
    Calculates used = 100 * (MemTotal - MemAvailable) / MemTotal.

    Args:
        threshold: Percentage threshold (0-100).
        operator: Comparison operator.

    Returns:
        True when condition is met.
    """
    mem = _read_meminfo()
    if mem is None:
        logger.debug("Memory sensor: cannot read /proc/meminfo")
        return False
    total = mem.get("MemTotal", 0)
    available = mem.get("MemAvailable", 0)
    if total <= 0:
        return False
    usage_pct = 100.0 * (total - available) / total
    triggered = _compare(usage_pct, threshold, operator)
    if triggered:
        logger.debug("Memory sensor triggered: %.1f%% %s %.0f%%",
                     usage_pct, operator, threshold)
    return triggered


@register_sensor("disk")
def check_disk(path: str = "/", threshold: float = 85, operator: str = ">") -> bool:
    """Check if disk usage at *path* exceeds threshold.

    Uses os.statvfs (Unix) or shutil.disk_usage (cross-platform fallback).

    Args:
        path: Filesystem path to check (default: "/").
        threshold: Percentage threshold (0-100).
        operator: Comparison operator.

    Returns:
        True when condition is met.
    """
    try:
        if hasattr(os, "statvfs"):
            st = os.statvfs(path)
            total = st.f_frsize * st.f_blocks
            free = st.f_frsize * st.f_bfree
        else:
            from shutil import disk_usage as _du
            du = _du(path)
            total = du.total
            free = du.free
        if total <= 0:
            return False
        used_pct = 100.0 * (total - free) / total
        triggered = _compare(used_pct, threshold, operator)
        if triggered:
            logger.debug("Disk sensor triggered (%s): %.1f%% %s %.0f%%",
                         path, used_pct, operator, threshold)
        return triggered
    except (OSError, IOError) as e:
        logger.debug("Disk sensor failed for '%s': %s", path, e)
        return False


@register_sensor("network")
def check_network(host: str = "8.8.8.8", count: int = 1, timeout: int = 2) -> bool:
    """Check network connectivity by pinging a host.

    Returns True when the ping FAILS (connectivity lost) — this is the
    "trigger" state that fires the cron job.

    Args:
        host: Host to ping (default: 8.8.8.8).
        count: Number of ping packets (default: 1).
        timeout: Ping timeout in seconds (default: 2).

    Returns:
        True when ping FAILS (no connectivity).
    """
    try:
        # Prefer ping with -c and -W flags (Linux)
        cmd = ["ping", "-c", str(count), "-W", str(timeout), host]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 2,
        )
        success = result.returncode == 0
        if not success:
            logger.debug("Network sensor triggered: ping to %s failed", host)
        return not success  # True = triggered = connectivity lost
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug("Network sensor: ping to %s failed with error: %s", host, e)
        return True  # Failed == triggered


@register_sensor("file")
def check_file(path: str, poll_interval: float = 0.0) -> bool:
    """Check if a file has changed since the last check.

    Stores the last mtime in a process-global cache. Returns True when
    the file's mtime is different from the cached value (file was modified).

    Args:
        path: Absolute path to the file or directory to watch.
        poll_interval: Minimum seconds between re-checks (default: 0 = every call).

    Returns:
        True when file has changed since last check.
    """
    check_path = Path(path).expanduser().resolve()
    if not check_path.exists():
        logger.debug("File sensor: path '%s' does not exist", path)
        return False

    # Use lstat to avoid following symlinks (watch the link itself)
    try:
        new_mtime = check_path.lstat().st_mtime
    except OSError:
        return False

    global _file_mtime_cache, _file_cache_lock
    with _file_cache_lock:
        last_mtime = _file_mtime_cache.get(str(check_path))
        if last_mtime is not None and abs(new_mtime - last_mtime) < 0.001:
            return False  # Not changed
        _file_mtime_cache[str(check_path)] = new_mtime
        logger.debug("File sensor triggered: '%s' changed (mtime: %s)",
                     check_path, new_mtime)
        return True


# Global file sensor cache
_file_mtime_cache: dict[str, float] = {}
_file_cache_lock = threading.Lock()


@register_sensor("process")
def check_process(name: str) -> bool:
    """Check if a process is NOT running.

    Uses pgrep (Linux/macOS). Returns True when the process is NOT found
    (trigger = process died / not running).

    Args:
        name: Process name to check (exact match with -x).

    Returns:
        True when process is NOT running (trigger state).
    """
    try:
        result = subprocess.run(
            ["pgrep", "-x", name],
            capture_output=True,
            timeout=5,
        )
        running = result.returncode == 0
        if not running:
            logger.debug("Process sensor triggered: '%s' is not running", name)
        return not running  # True = not running = trigger
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.debug("Process sensor: pgrep for '%s' failed: %s", name, e)
        return False  # Can't determine — assume running


# ---------------------------------------------------------------------------
# Utility: check if a job schedule uses sensors
# ---------------------------------------------------------------------------


def is_sensor_schedule(schedule: dict[str, Any]) -> bool:
    """Return True if the schedule dict describes a sensor-based trigger."""
    return schedule.get("kind") == "sensor"


def get_sensor_config(schedule: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Extract the sensor config from a schedule dict.

    Returns the sensor sub-dict, or None if not a sensor schedule.
    """
    if not is_sensor_schedule(schedule):
        return None
    return schedule.get("sensor", {})


def get_sensor_cooldown(schedule: dict[str, Any], default: int = 300) -> int:
    """Get the cooldown period for a sensor schedule in seconds.

    Args:
        schedule: The parsed schedule dict.
        default: Default cooldown if not specified (300s = 5min).

    Returns:
        Cooldown in seconds.
    """
    return int(schedule.get("cooldown", default))