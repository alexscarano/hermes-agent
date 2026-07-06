"""Tests for cron/sensors.py — sensor registry, comparisons, and individual checks."""

from __future__ import annotations

import subprocess

import pytest

import cron.sensors as sensors
from cron.sensors import (
    check_sensor,
    list_sensors,
    is_sensor_schedule,
    get_sensor_config,
    get_sensor_cooldown,
)


# =========================================================================
# Registry
# =========================================================================

class TestRegistry:
    def test_list_sensors_returns_all_builtin_types(self):
        assert list_sensors() == ["cpu", "disk", "file", "memory", "network", "process"]

    def test_check_sensor_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown sensor"):
            check_sensor("bogus", {})

    def test_check_sensor_swallows_exceptions_and_returns_false(self, monkeypatch):
        def _boom(**kwargs):
            raise RuntimeError("sensor exploded")

        monkeypatch.setattr(sensors, "_sensors", {**sensors._sensors, "cpu": _boom})
        assert check_sensor("cpu", {}) is False

    def test_check_sensor_passes_params_as_kwargs(self, monkeypatch):
        received = {}

        def _fake(threshold=1, operator=">"):
            received["threshold"] = threshold
            received["operator"] = operator
            return True

        monkeypatch.setattr(sensors, "_sensors", {**sensors._sensors, "cpu": _fake})
        assert check_sensor("cpu", {"threshold": 50, "operator": "<"}) is True
        assert received == {"threshold": 50, "operator": "<"}

    def test_check_sensor_none_params_defaults_to_empty(self, monkeypatch):
        monkeypatch.setattr(sensors, "_sensors", {**sensors._sensors, "cpu": lambda **kw: not kw})
        assert check_sensor("cpu", None) is True


# =========================================================================
# _compare
# =========================================================================

class TestCompare:
    def test_gt(self):
        assert sensors._compare(90, 80, ">") is True
        assert sensors._compare(70, 80, ">") is False

    def test_gte(self):
        assert sensors._compare(80, 80, ">=") is True
        assert sensors._compare(79, 80, ">=") is False

    def test_lt(self):
        assert sensors._compare(70, 80, "<") is True
        assert sensors._compare(90, 80, "<") is False

    def test_lte(self):
        assert sensors._compare(80, 80, "<=") is True
        assert sensors._compare(81, 80, "<=") is False

    def test_eq(self):
        assert sensors._compare(80.0001, 80, "==") is True
        assert sensors._compare(81, 80, "==") is False

    def test_neq(self):
        assert sensors._compare(81, 80, "!=") is True
        assert sensors._compare(80.0001, 80, "!=") is False

    def test_unknown_operator_defaults_to_gt(self):
        assert sensors._compare(90, 80, "?") is True
        assert sensors._compare(70, 80, "?") is False


# =========================================================================
# cpu sensor
# =========================================================================

class TestCpuSensor:
    def test_triggers_above_threshold(self, monkeypatch):
        monkeypatch.setattr(sensors, "_get_single_cpu_usage", lambda: 95.0)
        assert sensors.check_cpu(threshold=80) is True

    def test_does_not_trigger_below_threshold(self, monkeypatch):
        monkeypatch.setattr(sensors, "_get_single_cpu_usage", lambda: 50.0)
        assert sensors.check_cpu(threshold=80) is False

    def test_custom_operator(self, monkeypatch):
        monkeypatch.setattr(sensors, "_get_single_cpu_usage", lambda: 10.0)
        assert sensors.check_cpu(threshold=20, operator="<") is True

    def test_unreadable_proc_stat_returns_false(self, monkeypatch):
        monkeypatch.setattr(sensors, "_get_single_cpu_usage", lambda: None)
        assert sensors.check_cpu(threshold=80) is False


# =========================================================================
# memory sensor
# =========================================================================

class TestMemorySensor:
    def test_triggers_above_threshold(self, monkeypatch):
        # 90% used: MemTotal=1000, MemAvailable=100
        monkeypatch.setattr(
            sensors, "_read_meminfo", lambda: {"MemTotal": 1000, "MemAvailable": 100}
        )
        assert sensors.check_memory(threshold=80) is True

    def test_does_not_trigger_below_threshold(self, monkeypatch):
        # 10% used: MemTotal=1000, MemAvailable=900
        monkeypatch.setattr(
            sensors, "_read_meminfo", lambda: {"MemTotal": 1000, "MemAvailable": 900}
        )
        assert sensors.check_memory(threshold=80) is False

    def test_unreadable_meminfo_returns_false(self, monkeypatch):
        monkeypatch.setattr(sensors, "_read_meminfo", lambda: None)
        assert sensors.check_memory(threshold=80) is False

    def test_zero_total_returns_false(self, monkeypatch):
        monkeypatch.setattr(
            sensors, "_read_meminfo", lambda: {"MemTotal": 0, "MemAvailable": 0}
        )
        assert sensors.check_memory(threshold=80) is False


# =========================================================================
# disk sensor
# =========================================================================

class _FakeStatvfs:
    def __init__(self, f_frsize, f_blocks, f_bfree):
        self.f_frsize = f_frsize
        self.f_blocks = f_blocks
        self.f_bfree = f_bfree


class TestDiskSensor:
    def test_triggers_above_threshold(self, monkeypatch):
        # 90% used: total=1000, free=100
        monkeypatch.setattr(
            sensors.os, "statvfs", lambda path: _FakeStatvfs(1, 1000, 100)
        )
        assert sensors.check_disk(path="/", threshold=85) is True

    def test_does_not_trigger_below_threshold(self, monkeypatch):
        # 10% used: total=1000, free=900
        monkeypatch.setattr(
            sensors.os, "statvfs", lambda path: _FakeStatvfs(1, 1000, 900)
        )
        assert sensors.check_disk(path="/", threshold=85) is False

    def test_oserror_returns_false(self, monkeypatch):
        def _raise(path):
            raise OSError("no such path")

        monkeypatch.setattr(sensors.os, "statvfs", _raise)
        assert sensors.check_disk(path="/nope", threshold=85) is False


# =========================================================================
# network sensor
# =========================================================================

class _FakeCompletedProcess:
    def __init__(self, returncode):
        self.returncode = returncode


class TestNetworkSensor:
    def test_ping_success_does_not_trigger(self, monkeypatch):
        monkeypatch.setattr(
            sensors.subprocess, "run", lambda *a, **kw: _FakeCompletedProcess(0)
        )
        assert sensors.check_network(host="8.8.8.8") is False

    def test_ping_failure_triggers(self, monkeypatch):
        monkeypatch.setattr(
            sensors.subprocess, "run", lambda *a, **kw: _FakeCompletedProcess(1)
        )
        assert sensors.check_network(host="8.8.8.8") is True

    def test_ping_timeout_triggers(self, monkeypatch):
        def _raise(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="ping", timeout=2)

        monkeypatch.setattr(sensors.subprocess, "run", _raise)
        assert sensors.check_network(host="8.8.8.8") is True

    def test_missing_ping_binary_triggers(self, monkeypatch):
        def _raise(*a, **kw):
            raise FileNotFoundError("ping not found")

        monkeypatch.setattr(sensors.subprocess, "run", _raise)
        assert sensors.check_network(host="8.8.8.8") is True


# =========================================================================
# file sensor
# =========================================================================

class TestFileSensor:
    @pytest.fixture(autouse=True)
    def _clear_file_cache(self):
        sensors._file_mtime_cache.clear()
        yield
        sensors._file_mtime_cache.clear()

    def test_first_check_triggers(self, tmp_path):
        watched = tmp_path / "watch.txt"
        watched.write_text("v1")
        assert sensors.check_file(str(watched)) is True

    def test_unchanged_does_not_trigger_again(self, tmp_path):
        watched = tmp_path / "watch.txt"
        watched.write_text("v1")
        sensors.check_file(str(watched))
        assert sensors.check_file(str(watched)) is False

    def test_modified_file_triggers(self, tmp_path):
        import os
        import time

        watched = tmp_path / "watch.txt"
        watched.write_text("v1")
        sensors.check_file(str(watched))

        # Bump mtime forward so the change is unambiguous regardless of fs granularity.
        new_time = time.time() + 5
        os.utime(str(watched), (new_time, new_time))

        assert sensors.check_file(str(watched)) is True

    def test_nonexistent_path_returns_false(self, tmp_path):
        missing = tmp_path / "does_not_exist.txt"
        assert sensors.check_file(str(missing)) is False


# =========================================================================
# process sensor
# =========================================================================

class TestProcessSensor:
    def test_running_process_does_not_trigger(self, monkeypatch):
        monkeypatch.setattr(
            sensors.subprocess, "run", lambda *a, **kw: _FakeCompletedProcess(0)
        )
        assert sensors.check_process("nginx") is False

    def test_missing_process_triggers(self, monkeypatch):
        monkeypatch.setattr(
            sensors.subprocess, "run", lambda *a, **kw: _FakeCompletedProcess(1)
        )
        assert sensors.check_process("nginx") is True

    def test_pgrep_missing_binary_does_not_trigger(self, monkeypatch):
        def _raise(*a, **kw):
            raise FileNotFoundError("pgrep not found")

        monkeypatch.setattr(sensors.subprocess, "run", _raise)
        # Can't determine -> assume running -> no trigger
        assert sensors.check_process("nginx") is False


# =========================================================================
# schedule utility helpers
# =========================================================================

class TestScheduleUtilities:
    def test_is_sensor_schedule_true(self):
        assert is_sensor_schedule({"kind": "sensor"}) is True

    def test_is_sensor_schedule_false(self):
        assert is_sensor_schedule({"kind": "cron"}) is False
        assert is_sensor_schedule({}) is False

    def test_get_sensor_config_returns_sub_dict(self):
        schedule = {"kind": "sensor", "sensor": {"type": "cpu", "params": {"threshold": 80}}}
        assert get_sensor_config(schedule) == {"type": "cpu", "params": {"threshold": 80}}

    def test_get_sensor_config_none_for_non_sensor(self):
        assert get_sensor_config({"kind": "cron"}) is None

    def test_get_sensor_cooldown_default(self):
        assert get_sensor_cooldown({"kind": "sensor"}) == 300

    def test_get_sensor_cooldown_custom(self):
        assert get_sensor_cooldown({"kind": "sensor", "cooldown": 60}) == 60
