"""Tests for sensor-based cron schedules: parse_schedule, compute_next_run,
_get_due_jobs_locked, and mark_job_run integration."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from cron.jobs import (
    parse_schedule,
    compute_next_run,
    create_job,
    load_jobs,
    save_jobs,
    get_job,
    get_due_jobs,
    mark_job_run,
    _hermes_now,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Isolate cron job storage into a temp dir so tests don't stomp on real jobs."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


# =========================================================================
# parse_schedule — sensor formats
# =========================================================================

class TestParseScheduleSensor:
    def test_cpu(self):
        result = parse_schedule("sensor:cpu>80")
        assert result == {
            "kind": "sensor",
            "sensor": {"type": "cpu", "params": {"operator": ">", "threshold": 80}},
            "cooldown": 300,
            "display": "sensor:cpu>80",
        }

    def test_memory(self):
        result = parse_schedule("sensor:memory>=90")
        assert result["sensor"] == {
            "type": "memory",
            "params": {"operator": ">=", "threshold": 90},
        }

    def test_disk_with_path(self):
        result = parse_schedule("sensor:disk:/home>90")
        assert result["sensor"] == {
            "type": "disk",
            "params": {"path": "/home", "operator": ">", "threshold": 90},
        }

    def test_disk_root_path(self):
        result = parse_schedule("sensor:disk:/>85")
        assert result["sensor"] == {
            "type": "disk",
            "params": {"path": "/", "operator": ">", "threshold": 85},
        }

    def test_network(self):
        result = parse_schedule("sensor:network:8.8.8.8")
        assert result["kind"] == "sensor"
        assert result["sensor"] == {"type": "network", "params": {"host": "8.8.8.8"}}

    def test_file(self):
        result = parse_schedule("sensor:file:/tmp/watch")
        assert result["sensor"] == {"type": "file", "params": {"path": "/tmp/watch"}}

    def test_process(self):
        result = parse_schedule("sensor:process:nginx")
        assert result["sensor"] == {"type": "process", "params": {"name": "nginx"}}

    def test_default_cooldown_is_300(self):
        assert parse_schedule("sensor:cpu>80")["cooldown"] == 300

    def test_unknown_sensor_type_raises(self):
        with pytest.raises(ValueError, match="Unknown sensor type"):
            parse_schedule("sensor:bogus>80")

    def test_disk_without_path_raises(self):
        with pytest.raises(ValueError, match="disk sensor"):
            parse_schedule("sensor:disk")

    def test_disk_bare_type_with_operator_but_no_colon_is_unknown_type(self):
        # No ':' after "disk" means the whole "disk>80" is treated as the type
        # token, which isn't a recognized bare sensor name.
        with pytest.raises(ValueError, match="Unknown sensor type"):
            parse_schedule("sensor:disk>80")

    def test_cpu_without_operator_raises(self):
        with pytest.raises(ValueError):
            parse_schedule("sensor:cpu")

    def test_network_without_host_raises(self):
        with pytest.raises(ValueError, match="network sensor"):
            parse_schedule("sensor:network")

    def test_invalid_schedule_error_mentions_sensor(self):
        with pytest.raises(ValueError, match="Sensor:"):
            parse_schedule("not a valid schedule")


# =========================================================================
# compute_next_run — sensor kind has no predictable next run
# =========================================================================

class TestComputeNextRunSensor:
    def test_returns_none(self):
        schedule = {"kind": "sensor", "sensor": {"type": "cpu", "params": {}}, "cooldown": 300}
        assert compute_next_run(schedule) is None
        assert compute_next_run(schedule, last_run_at=_hermes_now().isoformat()) is None


# =========================================================================
# get_due_jobs / _get_due_jobs_locked — sensor evaluation + cooldown
# =========================================================================

class TestGetDueJobsSensor:
    def test_triggered_sensor_is_due(self, tmp_cron_dir, monkeypatch):
        monkeypatch.setattr("cron.jobs.check_sensor", lambda sensor_type, params: True)
        job = create_job(prompt="check cpu", schedule="sensor:cpu>80")

        due = get_due_jobs()

        assert [j["id"] for j in due] == [job["id"]]

    def test_not_triggered_sensor_is_not_due(self, tmp_cron_dir, monkeypatch):
        monkeypatch.setattr("cron.jobs.check_sensor", lambda sensor_type, params: False)
        create_job(prompt="check cpu", schedule="sensor:cpu>80")

        due = get_due_jobs()

        assert due == []

    def test_triggered_sensor_sets_next_run_at_to_cooldown(self, tmp_cron_dir, monkeypatch):
        monkeypatch.setattr("cron.jobs.check_sensor", lambda sensor_type, params: True)
        job = create_job(prompt="check cpu", schedule="sensor:cpu>80")

        get_due_jobs()

        updated = get_job(job["id"])
        next_dt = datetime.fromisoformat(updated["next_run_at"])
        now = _hermes_now()
        # cooldown default 300s
        assert now + timedelta(seconds=290) < next_dt <= now + timedelta(seconds=301)

    def test_cooldown_suppresses_reevaluation(self, tmp_cron_dir, monkeypatch):
        calls = []

        def _fake_check(sensor_type, params):
            calls.append(sensor_type)
            return True

        monkeypatch.setattr("cron.jobs.check_sensor", _fake_check)
        job = create_job(prompt="check cpu", schedule="sensor:cpu>80")

        # Simulate a very recent run — still within the 300s cooldown.
        jobs = load_jobs()
        jobs[0]["last_run_at"] = _hermes_now().isoformat()
        save_jobs(jobs)

        due = get_due_jobs()

        assert due == []
        assert calls == [], "sensor should not be evaluated while cooldown is active"

    def test_cooldown_expired_reevaluates(self, tmp_cron_dir, monkeypatch):
        monkeypatch.setattr("cron.jobs.check_sensor", lambda sensor_type, params: True)
        job = create_job(prompt="check cpu", schedule="sensor:cpu>80")

        # last_run_at far enough in the past that cooldown has lapsed.
        jobs = load_jobs()
        jobs[0]["last_run_at"] = (_hermes_now() - timedelta(seconds=301)).isoformat()
        save_jobs(jobs)

        due = get_due_jobs()

        assert [j["id"] for j in due] == [job["id"]]

    def test_disabled_sensor_job_never_evaluated(self, tmp_cron_dir, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "cron.jobs.check_sensor",
            lambda sensor_type, params: calls.append(1) or True,
        )
        job = create_job(prompt="check cpu", schedule="sensor:cpu>80")
        jobs = load_jobs()
        jobs[0]["enabled"] = False
        save_jobs(jobs)

        due = get_due_jobs()

        assert due == []
        assert calls == []

    def test_sensor_type_and_params_passed_through(self, tmp_cron_dir, monkeypatch):
        received = {}

        def _fake_check(sensor_type, params):
            received["type"] = sensor_type
            received["params"] = params
            return False

        monkeypatch.setattr("cron.jobs.check_sensor", _fake_check)
        create_job(prompt="watch disk", schedule="sensor:disk:/home>90")

        get_due_jobs()

        assert received == {"type": "disk", "params": {"path": "/home", "operator": ">", "threshold": 90}}

    def test_malformed_sensor_config_is_skipped_not_crashed(self, tmp_cron_dir, monkeypatch):
        """A hand-edited jobs.json entry missing "sensor"/"type" must not abort
        the whole tick's due-job scan for every other job (regression)."""
        good = create_job(prompt="check cpu", schedule="sensor:cpu>80")
        bad = create_job(prompt="broken", schedule="sensor:cpu>80")

        monkeypatch.setattr("cron.jobs.check_sensor", lambda sensor_type, params: True)

        jobs = load_jobs()
        for j in jobs:
            if j["id"] == bad["id"]:
                del j["schedule"]["sensor"]  # simulate corrupt/hand-edited store
        save_jobs(jobs)

        due = get_due_jobs()

        assert [j["id"] for j in due] == [good["id"]]

    def test_unregistered_sensor_type_is_skipped_not_crashed(self, tmp_cron_dir):
        """check_sensor() raises ValueError for an unregistered type — the scan
        must skip that job rather than propagate the exception (regression)."""
        good = create_job(prompt="check cpu", schedule="sensor:cpu>80")
        bad = create_job(prompt="ghost sensor", schedule="sensor:cpu>80")

        jobs = load_jobs()
        for j in jobs:
            if j["id"] == bad["id"]:
                j["schedule"]["sensor"]["type"] = "gpu_temp"  # never registered
        save_jobs(jobs)

        due = get_due_jobs()

        # good job still evaluated (real check_sensor, cpu>80 unlikely true in CI,
        # but the key assertion is that the scan didn't raise / abort).
        assert bad["id"] not in [j["id"] for j in due]


# =========================================================================
# mark_job_run — sensor jobs re-arm via cooldown, never "complete"
# =========================================================================

class TestMarkJobRunSensor:
    def test_success_sets_next_run_at_from_cooldown(self, tmp_cron_dir):
        job = create_job(prompt="check cpu", schedule="sensor:cpu>80")

        mark_job_run(job["id"], True)

        updated = get_job(job["id"])
        assert updated["next_run_at"] is not None
        next_dt = datetime.fromisoformat(updated["next_run_at"])
        now = _hermes_now()
        assert now + timedelta(seconds=290) < next_dt <= now + timedelta(seconds=301)
        # Sensor jobs are never auto-disabled/completed like exhausted one-shots.
        assert updated["enabled"] is True
        assert updated["state"] == "scheduled"

    def test_failure_still_rearms_via_cooldown(self, tmp_cron_dir):
        job = create_job(prompt="check cpu", schedule="sensor:cpu>80")

        mark_job_run(job["id"], False, error="boom")

        updated = get_job(job["id"])
        assert updated["next_run_at"] is not None
        assert updated["enabled"] is True
        assert updated["last_status"] == "error"

    def test_custom_cooldown_respected(self, tmp_cron_dir):
        job = create_job(prompt="check cpu", schedule="sensor:cpu>80")
        jobs = load_jobs()
        jobs[0]["schedule"]["cooldown"] = 60
        save_jobs(jobs)

        mark_job_run(job["id"], True)

        updated = get_job(job["id"])
        next_dt = datetime.fromisoformat(updated["next_run_at"])
        now = _hermes_now()
        assert now + timedelta(seconds=50) < next_dt <= now + timedelta(seconds=61)
