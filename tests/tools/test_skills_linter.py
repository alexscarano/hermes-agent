"""Tests for tools/skills_linter.py — skill syntax validation pipeline."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tools.skills_linter import (
    LintIssue,
    clear_lint_cache,
    format_lint_report,
    has_errors,
    lint_skill,
)

# ---------------------------------------------------------------------------
# Valid skill content (baseline — should produce zero issues)
# ---------------------------------------------------------------------------

VALID_SKILL = """\
---
name: my-test-skill
description: A valid skill for testing.
platforms: [linux, macos]
tags: [testing, linter]
version: "1.0.0"
author: Hermes Agent
license: MIT
---

# My Test Skill

Step 1: Do the thing.
Step 2: Do another thing.
"""

VALID_SKILL_WITH_METADATA = """\
---
name: nested-meta-skill
description: A skill with nested metadata.
platforms: [linux]
tags: [test]
metadata:
  hermes:
    tags: [nested]
    related_skills: [other-skill]
---

# Nested Meta Skill

Content here.
"""

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def test_lint_issue_dataclass():
    issue = LintIssue(severity="error", code="FM-001", message="Test issue", line=5)
    assert issue.severity == "error"
    assert issue.code == "FM-001"
    assert issue.message == "Test issue"
    assert issue.line == 5
    # Default line=None
    issue2 = LintIssue(severity="warning", code="W-001", message="Warning")
    assert issue2.line is None


def test_has_errors():
    assert has_errors([LintIssue("error", "E1", "err")]) is True
    assert has_errors([LintIssue("warning", "W1", "warn")]) is False
    assert has_errors([]) is False


def test_format_lint_report_empty():
    assert "no issues found" in format_lint_report([])


def test_format_lint_report_errors():
    issues = [
        LintIssue("error", "FM-001", "Missing frontmatter", line=1),
        LintIssue("warning", "BD-W01", "No heading"),
    ]
    report = format_lint_report(issues)
    assert "[SKILL LINT ERRORS — 1 issue(s)]" in report
    assert "[SKILL LINT WARNINGS — 1 issue(s)]" in report
    assert "FM-001" in report
    assert "BD-W01" in report


# ---------------------------------------------------------------------------
# FM-001: Frontmatter present
# ---------------------------------------------------------------------------


class TestFrontmatterPresent:
    def test_valid_frontmatter(self):
        issues = lint_skill(VALID_SKILL)
        fm_issues = [i for i in issues if i.code == "FM-001"]
        assert len(fm_issues) == 0

    def test_no_frontmatter(self):
        issues = lint_skill("# Just a heading\nNo frontmatter here.\n")
        fm_issues = [i for i in issues if i.code == "FM-001"]
        assert len(fm_issues) == 1
        assert fm_issues[0].severity == "error"

    def test_unclosed_frontmatter(self):
        issues = lint_skill("---\nname: test\ndescription: desc\nBody here.\n")
        fm_issues = [i for i in issues if i.code == "FM-001"]
        assert len(fm_issues) == 1
        assert fm_issues[0].severity == "error"


# ---------------------------------------------------------------------------
# FM-002: Required fields
# ---------------------------------------------------------------------------


class TestRequiredFields:
    def test_name_missing(self):
        content = """\
---
description: A skill without a name.
---

# Content
"""
        issues = lint_skill(content)
        req_issues = [i for i in issues if i.code == "FM-002"]
        assert any("name" in i.message for i in req_issues)

    def test_description_missing(self):
        content = """\
---
name: no-desc-skill
---

# Content
"""
        issues = lint_skill(content)
        req_issues = [i for i in issues if i.code == "FM-002"]
        assert any("description" in i.message for i in req_issues)

    def test_name_empty_string(self):
        content = """\
---
name: ""
description: A skill with empty name.
---

# Content
"""
        issues = lint_skill(content)
        req_issues = [i for i in issues if i.code == "FM-002"]
        assert any("name" in i.message for i in req_issues)

    def test_name_not_string(self):
        content = """\
---
name: true
description: A skill with boolean name.
---

# Content
"""
        issues = lint_skill(content)
        req_issues = [i for i in issues if i.code == "FM-002"]
        assert any("must be a string" in i.message for i in req_issues)


# ---------------------------------------------------------------------------
# FM-003: Name format
# ---------------------------------------------------------------------------


class TestNameFormat:
    def test_name_too_long(self):
        long_name = "a" * 65
        content = f"""\
---
name: {long_name}
description: A skill with a very long name.
---

# Content
"""
        issues = lint_skill(content)
        fm_issues = [i for i in issues if i.code == "FM-003"]
        assert len(fm_issues) == 1
        assert "exceeds" in fm_issues[0].message.lower()


# ---------------------------------------------------------------------------
# FM-004: Description length
# ---------------------------------------------------------------------------


class TestDescriptionLength:
    def test_description_too_long(self):
        long_desc = "x" * 1025
        content = f"""\
---
name: test-skill
description: {long_desc}
---

# Content
"""
        issues = lint_skill(content)
        fm_issues = [i for i in issues if i.code == "FM-004"]
        assert len(fm_issues) == 1
        assert "exceeds" in fm_issues[0].message.lower()


# ---------------------------------------------------------------------------
# FM-005: Platforms
# ---------------------------------------------------------------------------


class TestPlatforms:
    def test_valid_platforms(self):
        content = """\
---
name: p-skill
description: valid platforms
platforms: [linux, macos]
---

# Content
"""
        issues = lint_skill(content)
        assert len([i for i in issues if i.code == "FM-005"]) == 0

    def test_single_string_platform(self):
        content = """\
---
name: p-skill
description: single string platform
platforms: linux
---

# Content
"""
        issues = lint_skill(content)
        assert len([i for i in issues if i.code == "FM-005"]) == 0  # single string OK

    def test_invalid_platform(self):
        content = """\
---
name: p-skill
description: invalid platform
platforms: [ios, android]
---

# Content
"""
        issues = lint_skill(content)
        p_issues = [i for i in issues if i.code == "FM-005"]
        assert len(p_issues) == 2
        assert all(i.severity == "warning" for i in p_issues)

    def test_platform_not_string(self):
        content = """\
---
name: p-skill
description: platform not string
platforms: [123, true]
---

# Content
"""
        issues = lint_skill(content)
        p_issues = [i for i in issues if i.code == "FM-005"]
        assert len(p_issues) >= 1
        assert any(i.severity == "error" for i in p_issues)

    def test_platform_not_list(self):
        content = """\
---
name: p-skill
description: platform not list
platforms: true
---

# Content
"""
        issues = lint_skill(content)
        p_issues = [i for i in issues if i.code == "FM-005"]
        assert len(p_issues) >= 1


# ---------------------------------------------------------------------------
# FM-008: Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_metadata_not_dict(self):
        content = """\
---
name: m-skill
description: metadata as string
metadata: "just a string"
---

# Content
"""
        issues = lint_skill(content)
        m_issues = [i for i in issues if i.code == "FM-008"]
        assert len(m_issues) == 1
        assert m_issues[0].severity == "error"

    def test_metadata_as_list(self):
        content = """\
---
name: m-skill
description: metadata as list
metadata: [one, two]
---

# Content
"""
        issues = lint_skill(content)
        m_issues = [i for i in issues if i.code == "FM-008"]
        assert len(m_issues) == 1
        assert m_issues[0].severity == "error"

    def test_valid_metadata(self):
        content = """\
---
name: m-skill
description: valid metadata
metadata:
  key: value
  nested:
    inner: ok
---

# Content
"""
        issues = lint_skill(content)
        m_issues = [i for i in issues if i.code == "FM-008"]
        assert len(m_issues) == 0


# ---------------------------------------------------------------------------
# FM-009: Tags
# ---------------------------------------------------------------------------


class TestTags:
    def test_tags_single_string(self):
        content = """\
---
name: t-skill
description: tags as single string
tags: "testing"
---

# Content
"""
        issues = lint_skill(content)
        t_issues = [i for i in issues if i.code == "FM-009"]
        assert len(t_issues) == 1
        assert t_issues[0].severity == "warning"

    def test_tags_not_list(self):
        content = """\
---
name: t-skill
description: tags as number
tags: 42
---

# Content
"""
        issues = lint_skill(content)
        t_issues = [i for i in issues if i.code == "FM-009"]
        assert len(t_issues) >= 1

    def test_tags_valid(self):
        content = """\
---
name: t-skill
description: valid tags
tags: [devops, testing]
---

# Content
"""
        issues = lint_skill(content)
        t_issues = [i for i in issues if i.code == "FM-009"]
        assert len(t_issues) == 0


# ---------------------------------------------------------------------------
# FM-010: Related skills field
# ---------------------------------------------------------------------------


class TestRelatedSkillsField:
    def test_related_single_string(self):
        content = """\
---
name: r-skill
description: related as string
related_skills: other-skill
---

# Content
"""
        issues = lint_skill(content)
        r_issues = [i for i in issues if i.code == "FM-010"]
        assert len(r_issues) == 1
        assert r_issues[0].severity == "warning"

    def test_related_not_list(self):
        content = """\
---
name: r-skill
description: related as number
related_skills: 42
---

# Content
"""
        issues = lint_skill(content)
        r_issues = [i for i in issues if i.code == "FM-010"]
        assert len(r_issues) >= 1


# ---------------------------------------------------------------------------
# FM-011/FM-012/FM-013: Version, Author, License
# ---------------------------------------------------------------------------


class TestVersionAuthorLicense:
    def test_version_not_string(self):
        content = """\
---
name: v-skill
description: version as number
version: 1.0
---

# Content
"""
        issues = lint_skill(content)
        v_issues = [i for i in issues if i.code == "FM-011"]
        assert len(v_issues) == 1

    def test_author_not_string(self):
        content = """\
---
name: a-skill
description: author as list
author: [someone]
---

# Content
"""
        issues = lint_skill(content)
        a_issues = [i for i in issues if i.code == "FM-012"]
        assert len(a_issues) == 1

    def test_license_not_string(self):
        content = """\
---
name: l-skill
description: license as number
license: 123
---

# Content
"""
        issues = lint_skill(content)
        l_issues = [i for i in issues if i.code == "FM-013"]
        assert len(l_issues) == 1


# ---------------------------------------------------------------------------
# FM-014: Environments
# ---------------------------------------------------------------------------


class TestEnvironments:
    def test_valid_environment(self):
        content = """\
---
name: e-skill
description: valid environment
environments: [docker, kanban]
---

# Content
"""
        issues = lint_skill(content)
        assert len([i for i in issues if i.code == "FM-014"]) == 0

    def test_invalid_environment(self):
        content = """\
---
name: e-skill
description: invalid environment
environments: [production, staging]
---

# Content
"""
        issues = lint_skill(content)
        e_issues = [i for i in issues if i.code == "FM-014"]
        assert len(e_issues) == 2
        assert all(i.severity == "warning" for i in e_issues)

    def test_environment_not_list(self):
        content = """\
---
name: e-skill
description: environment as string
environments: docker
---

# Content
"""
        issues = lint_skill(content)
        e_issues = [i for i in issues if i.code == "FM-014"]
        # single string is OK
        assert len(e_issues) == 0


# ---------------------------------------------------------------------------
# FM-015: setup.collect_secrets
# ---------------------------------------------------------------------------


class TestSetupCollectSecrets:
    def test_valid_secrets(self):
        content = """\
---
name: s-skill
description: valid secrets
setup:
  collect_secrets:
    - env_var: API_KEY
      prompt: Enter your API key
---

# Content
"""
        issues = lint_skill(content)
        assert len([i for i in issues if i.code == "FM-015"]) == 0

    def test_secrets_not_list(self):
        content = """\
---
name: s-skill
description: secrets not list
setup:
  collect_secrets: "single string"
---

# Content
"""
        issues = lint_skill(content)
        s_issues = [i for i in issues if i.code == "FM-015"]
        assert len(s_issues) == 1

    def test_secret_missing_env_var(self):
        content = """\
---
name: s-skill
description: missing env_var
setup:
  collect_secrets:
    - prompt: Enter value
---

# Content
"""
        issues = lint_skill(content)
        s_issues = [i for i in issues if i.code == "FM-015"]
        assert len(s_issues) == 1
        assert "env_var" in s_issues[0].message


# ---------------------------------------------------------------------------
# BD-001: Body not empty
# ---------------------------------------------------------------------------


class TestBodyNotEmpty:
    def test_empty_body(self):
        content = """\
---
name: b-skill
description: empty body
---
"""
        issues = lint_skill(content)
        b_issues = [i for i in issues if i.code == "BD-001"]
        assert len(b_issues) == 1
        assert b_issues[0].severity == "error"

    def test_body_with_content(self):
        assert len([i for i in lint_skill(VALID_SKILL) if i.code == "BD-001"]) == 0


# ---------------------------------------------------------------------------
# BD-W01: Body heading
# ---------------------------------------------------------------------------


class TestBodyHeading:
    def test_no_heading(self):
        content = """\
---
name: h-skill
description: no heading
---

Just some text without a heading.
"""
        issues = lint_skill(content)
        h_issues = [i for i in issues if i.code == "BD-W01"]
        assert len(h_issues) == 1
        assert h_issues[0].severity == "warning"

    def test_with_heading(self):
        assert len([i for i in lint_skill(VALID_SKILL) if i.code == "BD-W01"]) == 0


# ---------------------------------------------------------------------------
# PLH-W01: Placeholders
# ---------------------------------------------------------------------------


class TestPlaceholders:
    def test_todo_detected(self):
        content = """\
---
name: p-skill
description: has TODO
---

# My Skill

TODO: implement this section.
"""
        issues = lint_skill(content)
        p_issues = [i for i in issues if i.code == "PLH-W01"]
        assert len(p_issues) >= 1

    def test_fixme_detected(self):
        content = """\
---
name: p-skill
description: has FIXME
---

# My Skill

FIXME: this is broken.
"""
        issues = lint_skill(content)
        p_issues = [i for i in issues if i.code == "PLH-W01"]
        assert len(p_issues) >= 1

    def test_no_placeholders(self):
        assert len([i for i in lint_skill(VALID_SKILL) if i.code == "PLH-W01"]) == 0


# ---------------------------------------------------------------------------
# BD-002: Name matches directory
# ---------------------------------------------------------------------------


class TestNameMatchesDirectory:
    def test_name_mismatch(self):
        content = """\
---
name: different-name
description: name does not match dir
---

# Content
"""
        issues = lint_skill(content, skill_name="actual-dir-name")
        name_issues = [i for i in issues if i.code == "BD-002"]
        assert len(name_issues) == 1
        assert name_issues[0].severity == "warning"

    def test_name_match(self):
        assert len([
            i for i in lint_skill(VALID_SKILL, skill_name="my-test-skill")
            if i.code == "BD-002"
        ]) == 0


# ---------------------------------------------------------------------------
# XRF-W01: Related skills exist on disk
# ---------------------------------------------------------------------------


class TestRelatedSkillsExist:
    def test_related_skill_not_found(self):
        content = """\
---
name: orphan-skill
description: has orphan related_skills
related_skills: [ghost-skill-xyz-123]
---

# Content
"""
        with patch("tools.skills_linter._skill_exists_on_disk", return_value=False):
            issues = lint_skill(content)
        xrf_issues = [i for i in issues if i.code == "XRF-W01"]
        assert len(xrf_issues) == 1
        assert xrf_issues[0].severity == "warning"

    def test_related_skill_found(self):
        content = """\
---
name: orphan-skill
description: has real related_skills
related_skills: [real-skill]
---

# Content
"""
        with patch("tools.skills_linter._skill_exists_on_disk", return_value=True):
            issues = lint_skill(content)
        assert len([i for i in issues if i.code == "XRF-W01"]) == 0


# ---------------------------------------------------------------------------
# LKF-W01: Linked file references
# ---------------------------------------------------------------------------


class TestLinkedFileReferences:
    def test_missing_linked_file(self, tmp_path):
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        # Create SKILL.md so the directory looks real
        (skill_dir / "SKILL.md").write_text("# placeholder")
        content = """\
---
name: test-skill
description: references missing file
---

# Test Skill

Load config with skill_view(name="test-skill", file_path="references/missing.yaml").
"""
        issues = lint_skill(content, skill_dir=skill_dir)
        lkf_issues = [i for i in issues if i.code == "LKF-W01"]
        assert len(lkf_issues) == 1
        assert lkf_issues[0].severity == "warning"

    def test_existing_linked_file(self, tmp_path):
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        ref_dir = skill_dir / "references"
        ref_dir.mkdir()
        (ref_dir / "config.yaml").write_text("# config content")
        (skill_dir / "SKILL.md").write_text("# placeholder")
        content = """\
---
name: test-skill
description: references existing file
---

# Test Skill

Load config with skill_view(name="test-skill", file_path="references/config.yaml").
"""
        issues = lint_skill(content, skill_dir=skill_dir)
        assert len([i for i in issues if i.code == "LKF-W01"]) == 0


# ---------------------------------------------------------------------------
# Integration tests — valid skills produce no errors
# ---------------------------------------------------------------------------


class TestValidSkills:
    def test_valid_baseline(self):
        issues = lint_skill(VALID_SKILL)
        assert len([i for i in issues if i.severity == "error"]) == 0

    def test_valid_with_metadata(self):
        issues = lint_skill(VALID_SKILL_WITH_METADATA)
        assert len([i for i in issues if i.severity == "error"]) == 0


# ---------------------------------------------------------------------------
# Cache 
# ---------------------------------------------------------------------------


class TestCache:
    def test_clear_cache(self):
        clear_lint_cache()
        # Cache should be empty after clear
        from tools.skills_linter import _skill_exists_cache
        assert len(_skill_exists_cache) == 0


# ---------------------------------------------------------------------------
# Prerequisites validation
# ---------------------------------------------------------------------------


class TestPrerequisites:
    def test_prerequisites_not_dict(self):
        content = """\
---
name: p-skill
description: prereqs not dict
prerequisites: "just a string"
---

# Content
"""
        issues = lint_skill(content)
        p_issues = [i for i in issues if i.code == "FM-006"]
        assert len(p_issues) == 1
        assert p_issues[0].severity == "error"


# ---------------------------------------------------------------------------
# Required env vars
# ---------------------------------------------------------------------------


class TestRequiredEnvVars:
    def test_required_not_list(self):
        content = """\
---
name: r-skill
description: reqs not list
required_environment_variables: "just a string"
---

# Content
"""
        issues = lint_skill(content)
        r_issues = [i for i in issues if i.code == "FM-016"]
        assert len(r_issues) == 1