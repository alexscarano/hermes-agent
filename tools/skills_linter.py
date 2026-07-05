#!/usr/bin/env python3
"""Skills Linter — Syntax validation pipeline for SKILL.md files.

Runs every time a skill is created or modified via ``skill_manage``.
Errors BLOCK the write; warnings let it through but are reported to the agent.

Schema validated:
  - YAML frontmatter field types and allowed values
  - Body structure (headings, content)
  - Cross-references (related_skills exist on disk)
  - Linked file references (skill_view targets exist)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Use the Hermes yaml loader (handles CSafeLoader fallback gracefully)
from agent.skill_utils import yaml_load as _hermes_yaml_load

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_PLATFORMS = frozenset({"macos", "linux", "windows"})
VALID_ENVIRONMENTS = frozenset({"kanban", "docker", "s6"})

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024

# ---------------------------------------------------------------------------
# LintIssue
# ---------------------------------------------------------------------------


@dataclass
class LintIssue:
    """A single lint finding.

    Attributes:
        severity: "error" (blocks write) or "warning" (reported, write allowed).
        code: Unique rule code (e.g. "FM-005").
        message: Human-readable description.
        line: Optional 1-based line number where the issue was detected.
    """

    severity: str  # "error" | "warning"
    code: str
    message: str
    line: Optional[int] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from a SKILL.md string.

    Returns (frontmatter_dict, body_content).
    On parse failure, frontmatter_dict is empty.
    """
    if not content.startswith("---"):
        return {}, content.strip()

    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return {}, content.strip()

    yaml_text = content[3: end_match.start() + 3]
    body = content[end_match.end() + 3:]

    try:
        parsed = _hermes_yaml_load(yaml_text)
        if isinstance(parsed, dict):
            return parsed, body.strip()
        return {}, body.strip()
    except Exception:
        return {}, body.strip()


def _find_line(content: str, marker: str) -> Optional[int]:
    """Return the 1-based line number where *marker* first appears in *content*."""
    for i, line in enumerate(content.split("\n"), start=1):
        if marker in line:
            return i
    return None


# ---------------------------------------------------------------------------
# Individual rule checkers
# ---------------------------------------------------------------------------


def _check_frontmatter_present(content: str) -> list[LintIssue]:
    """FM-001: Check that frontmatter markers exist."""
    issues: list[LintIssue] = []
    if not content.startswith("---"):
        issues.append(LintIssue(
            "error", "FM-001",
            "SKILL.md must start with YAML frontmatter (---).",
            line=1,
        ))
    else:
        end_match = re.search(r"\n---\s*\n", content[3:])
        if not end_match:
            issues.append(LintIssue(
                "error", "FM-001",
                "Frontmatter is not closed. Ensure a closing '---' line exists.",
                line=1,
            ))
    return issues


def _check_required_fields(fm: dict) -> list[LintIssue]:
    """FM-002: Check required top-level fields (name, description)."""
    issues: list[LintIssue] = []
    for field in ("name", "description"):
        if field not in fm or fm[field] is None:
            issues.append(LintIssue(
                "error", "FM-002",
                f"Frontmatter must include '{field}' field.",
            ))
        elif not isinstance(fm[field], str):
            issues.append(LintIssue(
                "error", "FM-002",
                f"Frontmatter '{field}' must be a string, got {type(fm[field]).__name__}.",
            ))
        elif isinstance(fm[field], str) and not fm[field].strip():
            issues.append(LintIssue(
                "error", "FM-002",
                f"Frontmatter '{field}' cannot be empty.",
            ))
    return issues


def _check_name_format(fm: dict) -> list[LintIssue]:
    """FM-003: Validate name length."""
    issues: list[LintIssue] = []
    name = fm.get("name", "")
    if not isinstance(name, str):
        return issues  # caught by _check_required_fields
    if len(name) > MAX_NAME_LENGTH:
        issues.append(LintIssue(
            "error", "FM-003",
            f"Skill name exceeds {MAX_NAME_LENGTH} characters ({len(name)}).",
        ))
    return issues


def _check_description_length(fm: dict) -> list[LintIssue]:
    """FM-004: Validate description length."""
    issues: list[LintIssue] = []
    desc = fm.get("description", "")
    if isinstance(desc, str) and len(desc) > MAX_DESCRIPTION_LENGTH:
        issues.append(LintIssue(
            "error", "FM-004",
            f"Description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(desc)}).",
        ))
    return issues


def _check_platforms(fm: dict) -> list[LintIssue]:
    """FM-005: Validate platforms field."""
    issues: list[LintIssue] = []
    platforms = fm.get("platforms")
    if platforms is None:
        return issues
    if isinstance(platforms, str):
        if platforms not in VALID_PLATFORMS:
            issues.append(LintIssue(
                "warning", "FM-005",
                f"Unknown platform '{platforms}'. Valid: {', '.join(sorted(VALID_PLATFORMS))}.",
            ))
        return issues
    if not isinstance(platforms, list):
        issues.append(LintIssue(
            "error", "FM-005",
            f"'platforms' must be a list of strings, got {type(platforms).__name__}.",
        ))
        return issues
    for p in platforms:
        if not isinstance(p, str):
            issues.append(LintIssue(
                "error", "FM-005",
                f"Each platform must be a string, got {type(p).__name__} ({p}).",
            ))
        elif p not in VALID_PLATFORMS:
            issues.append(LintIssue(
                "warning", "FM-005",
                f"Unknown platform '{p}'. Valid: {', '.join(sorted(VALID_PLATFORMS))}.",
            ))
    return issues


def _check_environments(fm: dict) -> list[LintIssue]:
    """FM-014: Validate environments field."""
    issues: list[LintIssue] = []
    envs = fm.get("environments")
    if envs is None:
        return issues
    if isinstance(envs, str):
        if envs not in VALID_ENVIRONMENTS:
            issues.append(LintIssue(
                "warning", "FM-014",
                f"Unknown environment '{envs}'. Valid: {', '.join(sorted(VALID_ENVIRONMENTS))}.",
            ))
        return issues
    if not isinstance(envs, list):
        issues.append(LintIssue(
            "error", "FM-014",
            f"'environments' must be a list of strings, got {type(envs).__name__}.",
        ))
        return issues
    for e in envs:
        if not isinstance(e, str):
            issues.append(LintIssue(
                "error", "FM-014",
                f"Each environment must be a string, got {type(e).__name__} ({e}).",
            ))
        elif e not in VALID_ENVIRONMENTS:
            issues.append(LintIssue(
                "warning", "FM-014",
                f"Unknown environment '{e}'. Valid: {', '.join(sorted(VALID_ENVIRONMENTS))}.",
            ))
    return issues


def _check_prerequisites(fm: dict) -> list[LintIssue]:
    """FM-006/FM-007: Validate prerequisites structure."""
    issues: list[LintIssue] = []
    prereqs = fm.get("prerequisites")
    if prereqs is None:
        return issues
    if not isinstance(prereqs, dict):
        issues.append(LintIssue(
            "error", "FM-006",
            f"'prerequisites' must be a mapping (dict), got {type(prereqs).__name__}.",
        ))
        return issues
    for key, label in (("env_vars", "FM-006"), ("commands", "FM-007")):
        val = prereqs.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            continue  # single string OK, normalized later
        if isinstance(val, list):
            for item in val:
                if not isinstance(item, str):
                    issues.append(LintIssue(
                        "error", label,
                        f"Each item in 'prerequisites.{key}' must be a string, got {type(item).__name__}.",
                    ))
        else:
            issues.append(LintIssue(
                "error", label,
                f"'prerequisites.{key}' must be a string or list of strings, got {type(val).__name__}.",
            ))
    return issues


def _check_metadata(fm: dict) -> list[LintIssue]:
    """FM-008: Validate metadata field is a mapping."""
    issues: list[LintIssue] = []
    metadata = fm.get("metadata")
    if metadata is None:
        return issues
    if not isinstance(metadata, dict):
        issues.append(LintIssue(
            "error", "FM-008",
            f"'metadata' must be a mapping (dict), got {type(metadata).__name__}.",
        ))
    return issues


def _check_tags(fm: dict) -> list[LintIssue]:
    """FM-009: Validate tags field."""
    issues: list[LintIssue] = []
    tags = fm.get("tags")
    if tags is None:
        return issues
    if isinstance(tags, str):
        issues.append(LintIssue(
            "warning", "FM-009",
            "'tags' should be a list of strings, not a single string.",
        ))
        return issues
    if not isinstance(tags, list):
        issues.append(LintIssue(
            "error", "FM-009",
            f"'tags' must be a list of strings, got {type(tags).__name__}.",
        ))
        return issues
    for t in tags:
        if not isinstance(t, str):
            issues.append(LintIssue(
                "error", "FM-009",
                f"Each tag must be a string, got {type(t).__name__} ({t}).",
            ))
    return issues


def _check_related_skills_field(fm: dict) -> list[LintIssue]:
    """FM-010: Validate related_skills field type."""
    issues: list[LintIssue] = []
    related = fm.get("related_skills")
    if related is None:
        return issues
    if isinstance(related, str):
        issues.append(LintIssue(
            "warning", "FM-010",
            "'related_skills' should be a list of strings, not a single string.",
        ))
        return issues
    if not isinstance(related, list):
        issues.append(LintIssue(
            "error", "FM-010",
            f"'related_skills' must be a list of strings, got {type(related).__name__}.",
        ))
        return issues
    for rs in related:
        if not isinstance(rs, str):
            issues.append(LintIssue(
                "error", "FM-010",
                f"Each related_skill must be a string, got {type(rs).__name__} ({rs}).",
            ))
    return issues


def _check_version_author_license(fm: dict) -> list[LintIssue]:
    """FM-011/FM-012/FM-013: Validate version, author, license are strings."""
    issues: list[LintIssue] = []
    for field, code in (("version", "FM-011"), ("author", "FM-012"), ("license", "FM-013")):
        val = fm.get(field)
        if val is None:
            continue
        if not isinstance(val, str):
            issues.append(LintIssue(
                "error", code,
                f"'{field}' must be a string, got {type(val).__name__}.",
            ))
    return issues


def _check_body_not_empty(body: str, content: str) -> list[LintIssue]:
    """BD-001: Body must have content after frontmatter."""
    issues: list[LintIssue] = []
    if not body:
        end_match = re.search(r"\n---\s*\n", content[3:]) if content.startswith("---") else None
        line = (content[:end_match.end() + 3].count("\n") + 2) if end_match else None
        issues.append(LintIssue(
            "error", "BD-001",
            "SKILL.md must have content after the frontmatter (instructions, procedures, etc.).",
            line=line,
        ))
    return issues


def _check_body_has_heading(body: str) -> list[LintIssue]:
    """BD-W01: Body should have a top-level heading."""
    issues: list[LintIssue] = []
    if body and not re.search(r"^#\s+", body, re.MULTILINE):
        issues.append(LintIssue(
            "warning", "BD-W01",
            "Body should start with a top-level markdown heading (# Title).",
        ))
    return issues


def _check_placeholders(body: str) -> list[LintIssue]:
    """PLH-W01: Detect placeholder/TODO content."""
    issues: list[LintIssue] = []
    patterns = [
        (r"\bTODO\b", "Contains 'TODO' placeholder text."),
        (r"\bFIXME\b", "Contains 'FIXME' placeholder text."),
        (r"\blorem ipsum\b", "Contains placeholder text ('lorem ipsum')."),
        (r"\bplaceholder\b", "Contains placeholder text."),
    ]
    for pattern, msg in patterns:
        m = re.search(pattern, body, re.IGNORECASE)
        if m:
            line = _find_line(body, m.group())
            issues.append(LintIssue("warning", "PLH-W01", msg, line=line))
    return issues


def _check_related_skills_exist(fm: dict) -> list[LintIssue]:
    """XRF-W01: Check related_skills reference skills that exist on disk."""
    issues: list[LintIssue] = []
    related = fm.get("related_skills", [])
    if isinstance(related, str):
        related = [related]
    if not isinstance(related, list):
        return issues
    for rs in related:
        if not isinstance(rs, str) or not rs.strip():
            continue
        found = _skill_exists_on_disk(rs.strip())
        if found is False:
            issues.append(LintIssue(
                "warning", "XRF-W01",
                f"related_skill '{rs}' not found in any skill directory. "
                "Create it first or remove the reference.",
            ))
    return issues


def _check_linked_file_references(body: str, skill_dir: Optional[Path]) -> list[LintIssue]:
    """LKF-W01: Check skill_view(file_path=...) references point to real files."""
    issues: list[LintIssue] = []
    if not skill_dir or not skill_dir.exists():
        return issues
    pattern = r"skill_view\([^)]*file_path\s*=\s*['\"]([^'\"]+)['\"]"
    for m in re.finditer(pattern, body):
        file_path = m.group(1)
        target = skill_dir / file_path
        if not target.exists():
            issues.append(LintIssue(
                "warning", "LKF-W01",
                f"Linked file '{file_path}' not found in skill directory "
                f"({skill_dir.name}). Create it or fix the path.",
                line=_find_line(body, m.group()),
            ))
    return issues


def _check_name_matches_directory(fm: dict, skill_name: Optional[str]) -> list[LintIssue]:
    """BD-002: Warn if frontmatter name differs from directory name."""
    issues: list[LintIssue] = []
    if not skill_name:
        return issues
    fm_name = fm.get("name", "")
    if isinstance(fm_name, str) and fm_name.strip() and \
            fm_name.strip().lower() != skill_name.lower():
        issues.append(LintIssue(
            "warning", "BD-002",
            f"Frontmatter name '{fm_name}' differs from directory name "
            f"'{skill_name}'. Consider keeping them consistent.",
        ))
    return issues


def _check_setup_collect_secrets(fm: dict) -> list[LintIssue]:
    """FM-015: Validate setup.collect_secrets entries."""
    issues: list[LintIssue] = []
    setup = fm.get("setup")
    if not isinstance(setup, dict):
        return issues
    secrets = setup.get("collect_secrets")
    if secrets is None:
        return issues
    if not isinstance(secrets, list):
        issues.append(LintIssue(
            "error", "FM-015",
            f"'setup.collect_secrets' must be a list of mappings, "
            f"got {type(secrets).__name__}.",
        ))
        return issues
    for i, entry in enumerate(secrets):
        if not isinstance(entry, dict):
            issues.append(LintIssue(
                "error", "FM-015",
                f"Each entry in 'setup.collect_secrets' must be a mapping, "
                f"got {type(entry).__name__} at index {i}.",
            ))
            continue
        if "env_var" not in entry or not isinstance(entry.get("env_var"), str):
            issues.append(LintIssue(
                "error", "FM-015",
                f"Each 'setup.collect_secrets' entry must have a string 'env_var' "
                f"field (missing or invalid at index {i}).",
            ))
    return issues


def _check_required_env_vars(fm: dict) -> list[LintIssue]:
    """FM-016: Validate required_environment_variables entries."""
    issues: list[LintIssue] = []
    reqs = fm.get("required_environment_variables")
    if reqs is None:
        return issues
    if not isinstance(reqs, list):
        issues.append(LintIssue(
            "error", "FM-016",
            f"'required_environment_variables' must be a list of mappings, "
            f"got {type(reqs).__name__}.",
        ))
        return issues
    for i, entry in enumerate(reqs):
        if not isinstance(entry, dict):
            issues.append(LintIssue(
                "error", "FM-016",
                f"Each entry in 'required_environment_variables' must be a mapping, "
                f"got {type(entry).__name__} at index {i}.",
            ))
            continue
        if "name" not in entry or not isinstance(entry.get("name"), str):
            issues.append(LintIssue(
                "warning", "FM-016",
                f"Each 'required_environment_variables' entry should have a string "
                f"'name' field (missing or invalid at index {i}).",
            ))
    return issues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lint_skill(
    content: str,
    skill_dir: Optional[Path] = None,
    skill_name: Optional[str] = None,
) -> list[LintIssue]:
    """Run all lint rules against skill content.

    Args:
        content: Full SKILL.md text (frontmatter + body).
        skill_dir: Path to the skill directory (for linked file checks).
        skill_name: Expected skill name (for name/directory consistency).

    Returns:
        List of LintIssue. Empty list = clean.
    """
    fm, body = _parse_frontmatter(content)

    issues: list[LintIssue] = []

    # Schema/frontmatter checks
    issues.extend(_check_frontmatter_present(content))
    issues.extend(_check_required_fields(fm))
    issues.extend(_check_name_format(fm))
    issues.extend(_check_description_length(fm))
    issues.extend(_check_platforms(fm))
    issues.extend(_check_environments(fm))
    issues.extend(_check_prerequisites(fm))
    issues.extend(_check_metadata(fm))
    issues.extend(_check_tags(fm))
    issues.extend(_check_related_skills_field(fm))
    issues.extend(_check_version_author_license(fm))
    issues.extend(_check_setup_collect_secrets(fm))
    issues.extend(_check_required_env_vars(fm))

    # Body checks
    issues.extend(_check_body_not_empty(body, content))
    issues.extend(_check_body_has_heading(body))
    issues.extend(_check_placeholders(body))

    # Cross-reference checks
    issues.extend(_check_related_skills_exist(fm))
    issues.extend(_check_linked_file_references(body, skill_dir))
    issues.extend(_check_name_matches_directory(fm, skill_name))

    return issues


def has_errors(issues: list[LintIssue]) -> bool:
    """Return True if any issue has severity 'error'."""
    return any(i.severity == "error" for i in issues)


def format_lint_report(issues: list[LintIssue]) -> str:
    """Format lint issues into a human-readable report."""
    if not issues:
        return "Skills linter: no issues found."

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    parts: list[str] = []
    if errors:
        parts.append(f"[SKILL LINT ERRORS — {len(errors)} issue(s)]")
        for issue in errors:
            line_str = f" (line {issue.line})" if issue.line else ""
            parts.append(f"  [{issue.code}]{line_str} {issue.message}")
    if warnings:
        parts.append(f"[SKILL LINT WARNINGS — {len(warnings)} issue(s)]")
        for issue in warnings:
            line_str = f" (line {issue.line})" if issue.line else ""
            parts.append(f"  [{issue.code}]{line_str} {issue.message}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Disk search helper
# ---------------------------------------------------------------------------

_skill_exists_cache: dict[str, Optional[bool]] = {}


def _skill_exists_on_disk(skill_name: str) -> Optional[bool]:
    """Check if a skill named *skill_name* exists in any skill directory.

    Results are cached per process.
    """
    if skill_name in _skill_exists_cache:
        return _skill_exists_cache[skill_name]
    try:
        from hermes_constants import get_hermes_home
        skills_root = get_hermes_home() / "skills"
        if not skills_root.is_dir():
            _skill_exists_cache[skill_name] = False
            return False
        for entry in skills_root.rglob("SKILL.md"):
            if entry.parent.name == skill_name:
                _skill_exists_cache[skill_name] = True
                return True
        # Check external skill dirs too
        try:
            from agent.skill_utils import get_external_skills_dirs
            for ext_root in get_external_skills_dirs():
                if ext_root.is_dir():
                    for entry in ext_root.rglob("SKILL.md"):
                        if entry.parent.name == skill_name:
                            _skill_exists_cache[skill_name] = True
                            return True
        except Exception:
            pass
        _skill_exists_cache[skill_name] = False
        return False
    except Exception:
        _skill_exists_cache[skill_name] = None
        return None


def clear_lint_cache() -> None:
    """Clear all internal caches (for testing)."""
    _skill_exists_cache.clear()