"""Tests for tools/skills_sast.py — SAST security scanner for skills."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tools.skills_sast import (
    SastIssue,
    scan_skill,
    has_errors,
    format_report,
)


# ---------------------------------------------------------------------------
# Valid skill (baseline — clean)
# ---------------------------------------------------------------------------

CLEAN_SKILL = """\
---
name: clean-skill
description: A perfectly safe skill.
---

# Clean Skill

Use os.getenv to read credentials:
```python
import os
api_key = os.getenv("API_KEY")
db_pass = os.getenv("DB_PASSWORD")
```

Safe operations:
```python
import shutil
path = "/tmp/safe"
if path.startswith("/tmp/"):
    shutil.rmtree(path)
```
"""


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def test_sast_issue_dataclass():
    issue = SastIssue(severity="critical", code="CRED-001", message="Key found", line=5)
    assert issue.severity == "critical"
    assert issue.code == "CRED-001"
    assert issue.line == 5
    assert issue.file is None


def test_has_errors():
    assert has_errors([SastIssue("critical", "C1", "err")]) is True
    assert has_errors([SastIssue("high", "H1", "err")]) is True
    assert has_errors([SastIssue("medium", "M1", "warn")]) is False
    assert has_errors([SastIssue("low", "L1", "warn")]) is False
    assert has_errors([]) is False


def test_format_report():
    issues = [
        SastIssue("critical", "CRED-001", "API key hardcoded", line=3),
        SastIssue("low", "NET-001", "HTTP URL used", line=10),
    ]
    report = format_report(issues)
    assert "CRITICAL" in report.upper() or "CRED-001" in report
    assert "NET-001" in report


# ---------------------------------------------------------------------------
# CRED: Hardcoded credentials
# ---------------------------------------------------------------------------


class TestCRED:
    def test_api_key_sk_detected(self):
        content = """\
---
name: bad-skill
description: has api key
---

```python
api_key = "sk-proj-abc123def456"
```
"""
        issues = scan_skill(content)
        cred = [i for i in issues if i.code.startswith("CRED")]
        assert len(cred) >= 1

    def test_password_detected(self):
        content = """\
---
name: bad-skill
description: has password
---

```python
password = "supersecret123"
```
"""
        issues = scan_skill(content)
        cred = [i for i in issues if i.code.startswith("CRED")]
        assert len(cred) >= 1

    def test_url_with_credentials_detected(self):
        content = """\
---
name: bad-skill
description: has url creds
---

```
https://user:pass@example.com/db
```
"""
        issues = scan_skill(content)
        cred = [i for i in issues if i.code.startswith("CRED")]
        assert len(cred) >= 1

    def test_os_getenv_not_flagged(self):
        """os.getenv should NOT trigger false positive."""
        content = """\
---
name: safe-skill
description: uses env vars properly
---

```python
import os
api_key = os.getenv("API_KEY")
db_password = os.getenv("DB_PASSWORD")
```
"""
        issues = scan_skill(content)
        cred = [i for i in issues if i.code.startswith("CRED")]
        critical_cred = [i for i in cred if i.severity in ("critical", "high")]
        assert len(critical_cred) == 0


# ---------------------------------------------------------------------------
# INJ: Command injection
# ---------------------------------------------------------------------------


class TestINJ:
    def test_os_system_detected(self):
        content = """\
---
name: bad-skill
description: uses os.system
---

```python
import os
os.system("rm -rf " + user_input)
```
"""
        issues = scan_skill(content)
        inj = [i for i in issues if i.code.startswith("INJ")]
        assert len(inj) >= 1

    def test_eval_detected(self):
        content = """\
---
name: bad-skill
description: uses eval
---

```python
eval(f"process({user_data})")
```
"""
        issues = scan_skill(content)
        inj = [i for i in issues if i.code.startswith("INJ")]
        assert len(inj) >= 1


# ---------------------------------------------------------------------------
# FILE: Unsafe file operations
# ---------------------------------------------------------------------------


class TestFILE:
    def test_shutil_rmtree_root_detected(self):
        content = """\
---
name: bad-skill
description: rmtree on /
---

```python
import shutil
shutil.rmtree("/some/absolute/path")
```
"""
        issues = scan_skill(content)
        file_issues = [i for i in issues if i.code.startswith("FILE")]
        assert len(file_issues) >= 1

    def test_write_to_system_path_detected(self):
        content = """\
---
name: bad-skill
description: write to /etc
---

```python
with open("/etc/config.ini", "w") as f:
    f.write("data")
```
"""
        issues = scan_skill(content)
        file_issues = [i for i in issues if i.code.startswith("FILE")]
        assert len(file_issues) >= 1


# ---------------------------------------------------------------------------
# IMP: Dangerous imports
# ---------------------------------------------------------------------------


class TestIMP:
    def test_pickle_import_detected(self):
        content = """\
---
name: bad-skill
description: imports pickle
---

```python
import pickle
data = pickle.loads(raw)
```
"""
        issues = scan_skill(content)
        imp = [i for i in issues if i.code.startswith("IMP")]
        assert len(imp) >= 1

    def test_ctypes_import_detected(self):
        content = """\
---
name: bad-skill
description: imports ctypes
---

```python
import ctypes
libc = ctypes.CDLL("libc.so.6")
```
"""
        issues = scan_skill(content)
        imp = [i for i in issues if i.code.startswith("IMP")]
        assert len(imp) >= 1


# ---------------------------------------------------------------------------
# NET: Network insecurity
# ---------------------------------------------------------------------------


class TestNET:
    def test_verify_false_detected(self):
        content = """\
---
name: bad-skill
description: verify=False
---

```python
import requests
r = requests.get("https://example.com", verify=False)
```
"""
        issues = scan_skill(content)
        net = [i for i in issues if i.code.startswith("NET")]
        assert len(net) >= 1


# ---------------------------------------------------------------------------
# Clean skill
# ---------------------------------------------------------------------------


class TestClean:
    def test_clean_skill_no_errors(self):
        issues = scan_skill(CLEAN_SKILL)
        critical_or_high = [i for i in issues if i.severity in ("critical", "high")]
        assert len(critical_or_high) == 0, f"Clean skill has issues: {issues}"


# ---------------------------------------------------------------------------
# Integration: SAST + skill_manager_tool
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_sast_blocks_create_with_credentials(self, tmp_path):
        from unittest.mock import patch

        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
             patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
            from tools.skill_manager_tool import _create_skill

            bad_content = """\
---
name: leaky-skill
description: has hardcoded key
---

```python
api_key = "sk-abc123def456"
```
"""
            result = _create_skill("leaky-skill", bad_content)
            assert result["success"] is False, f"Should be blocked: {result}"
            assert "SAST" in result.get("error", "").upper() or \
                   "blocked" in result.get("error", "").lower() or \
                   "cred" in result.get("error", "").lower(), \
                   f"Error should mention SAST/cred: {result}"

    def test_clean_skill_passes_sast(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
             patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
            from tools.skill_manager_tool import _create_skill
            result = _create_skill("safe-skill", CLEAN_SKILL)
            assert result["success"] is True, f"Clean skill blocked: {result}"