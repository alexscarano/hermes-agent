#!/usr/bin/env python3
"""Skills SAST — Static Application Security Testing for SKILL.md content.

Scans skill content for security vulnerabilities including hardcoded
credentials, command injection risks, unsafe file operations, dangerous
imports, and network insecurity.

Errors (critical/high severity) BLOCK the skill write; medium/low are
reported to the agent but do not block.

Severity guide:
  critical — Credential leak (API keys, passwords in plain text)
  high     — Command injection / dangerous eval / SSL bypass
  medium   — Unsafe file operations, dangerous imports
  low      — HTTP URLs for sensitive operations
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SastIssue
# ---------------------------------------------------------------------------


@dataclass
class SastIssue:
    """A single SAST finding.

    Attributes:
        severity: "critical" | "high" | "medium" | "low"
        code: Unique rule code (e.g. "CRED-001").
        message: Human-readable description.
        line: Optional 1-based line number where the issue was detected.
        file: Optional filename (e.g. "SKILL.md", or a script path).
    """

    severity: str  # "critical" | "high" | "medium" | "low"
    code: str
    message: str
    line: Optional[int] = None
    file: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_line(content: str, marker: str) -> Optional[int]:
    """Return the 1-based line number where *marker* first appears."""
    for i, line in enumerate(content.split("\n"), start=1):
        if marker in line:
            return i
    return None


def _find_line_re(content: str, pattern: re.Pattern) -> Optional[int]:
    """Return the 1-based line number matching the regex pattern."""
    for i, line in enumerate(content.split("\n"), start=1):
        if pattern.search(line):
            return i
    return None


# ---------------------------------------------------------------------------
# Rule checkers
# ---------------------------------------------------------------------------


# ── Hardcoded credentials ───────────────────────────────────────────────────


def _check_cred_001(content: str) -> list[SastIssue]:
    """CRED-001: API keys in plain text.

    Detects:
      - sk-... tokens (OpenAI-style)
      - api_key="..." or api_key='...'
      - tokens > 20 alphanumeric chars inside quotes
    """
    issues: list[SastIssue] = []
    lines = content.split("\n")

    # Pattern A: sk-... style API key tokens (common for OpenAI, Anthropic)
    for i, line in enumerate(lines, start=1):
        # OpenAI-style: sk- followed by alphanumeric
        if re.search(r'sk-[a-zA-Z0-9]{20,}', line):
            # Exclude examples/placeholders
            if not re.search(r'(example|placeholder|your_key_here|sk-your)', line, re.IGNORECASE):
                issues.append(SastIssue(
                    "critical", "CRED-001",
                    "Possible API key token detected (sk-...). Remove hardcoded keys and use environment variables instead.",
                    line=i,
                ))

    # Pattern B: api_key = "literal_value" (not os.getenv or lookup)
    for i, line in enumerate(lines, start=1):
        m = re.search(r'(?:api_key|api-key|apikey)\s*[:=]\s*[\'"]([^\'"]+)[\'"]', line, re.IGNORECASE)
        if m:
            val = m.group(1)
            # Exclude variable references and examples
            if not re.search(r'\$\{?\w+}?|os\.getenv|getenv|\.env|example|placeholder', val, re.IGNORECASE):
                issues.append(SastIssue(
                    "critical", "CRED-001",
                    f"Hardcoded API key found: `{m.group(0)}`. Use an environment variable (os.getenv) instead.",
                    line=i,
                ))

    # Pattern C: Long alphanumeric tokens in quotes that look like keys
    for i, line in enumerate(lines, start=1):
        # Match any quoted string > 20 chars that is mostly alphanumeric
        m = re.search(r'[\'"]([a-zA-Z0-9_-]{20,})[\'"]', line)
        if m:
            val = m.group(1)
            # Skip obvious non-secrets: URLs, file paths, examples, version strings, UUIDs
            if re.search(r'^[a-f0-9]{32}$', val, re.IGNORECASE):
                continue  # MD5 hash, might be legitimate
            if re.search(r'^\d{4}-\d{4}-\d{4}-\d{4}', val):
                continue  # formatted string
            # Check if it looks like a token (mix of uppercase, lowercase, digits, underscores, hyphens)
            # and is not clearly an example/placeholder
            if not re.search(r'(example|placeholder|your_token|changeme|test_token)', line, re.IGNORECASE):
                alpha_ratio = sum(1 for c in val if c.isalpha()) / len(val)
                if alpha_ratio > 0.4 and len(val) > 24:
                    issues.append(SastIssue(
                        "critical", "CRED-001",
                        f"Possible hardcoded credential token ({len(val)} chars). Use an environment variable instead.",
                        line=i,
                    ))

    return issues


def _check_cred_002(content: str) -> list[SastIssue]:
    """CRED-002: Passwords / credentials in URLs or inline password assignments."""
    issues: list[SastIssue] = []
    lines = content.split("\n")

    for i, line in enumerate(lines, start=1):
        # Pattern A: password = "literal" or password: "literal"
        m = re.search(r'password\s*[:=]\s*[\'"]([^\'"]+)[\'"]', line, re.IGNORECASE)
        if m:
            val = m.group(1)
            if not re.search(r'\$\{?\w+}?|os\.getenv|getenv|\.env|example|placeholder', val, re.IGNORECASE):
                issues.append(SastIssue(
                    "critical", "CRED-002",
                    "Hardcoded password detected. Use an environment variable or secrets management instead.",
                    line=i,
                ))

        # Pattern B: URLs with inline credentials (https://user:pass@host)
        m = re.search(r'https?://([^:]+):([^@]+)@', line)
        if m:
            user, pwd = m.group(1), m.group(2)
            if not re.search(r'(example|placeholder|your_?\w+_?name|your_?\w+_?pass)', user, re.IGNORECASE):
                issues.append(SastIssue(
                    "critical", "CRED-002",
                    f"URL with inline credentials detected (user:{'*' * len(pwd)}@host). Store credentials in environment variables.",
                    line=i,
                ))

    return issues


def _check_cred_003(content: str) -> list[SastIssue]:
    """CRED-003: Environment variable assignments with hardcoded values instead of os.getenv."""
    issues: list[SastIssue] = []
    lines = content.split("\n")

    # Common env var names that should use os.getenv
    sensitive_env_patterns = [
        r'API_KEY',
        r'TOKEN',
        r'SECRET',
        r'PASSWORD',
        r'PASSWD',
        r'CREDENTIAL',
        r'API_SECRET',
        r'ACCESS_KEY',
        r'SECRET_KEY',
        r'AUTH_TOKEN',
        r'BEARER',
        r'OPENAI',
        r'ANTHROPIC',
    ]

    for i, line in enumerate(lines, start=1):
        for env_pattern in sensitive_env_patterns:
            # Match: SOME_VAR = "literal-value"  (not os.getenv)
            m = re.search(
                rf'({env_pattern})\s*[:=]\s*[\'"]([^\'"]+)[\'"]',
                line,
                re.IGNORECASE,
            )
            if m:
                var_name = m.group(1)
                value = m.group(2)
                # Ensure it's a literal, not an os.getenv call
                if not re.search(r'\$\{?\w+}?|os\.getenv|getenv|environ\.get|\.env|example|placeholder|your_', value, re.IGNORECASE):
                    # Only flag if the value doesn't look like a reference
                    if not re.search(r'^\$\{?\w+', value):
                        # Allow short placeholder values
                        if len(value) > 3:
                            issues.append(SastIssue(
                                "high", "CRED-003",
                                f"Environment variable '{var_name}' assigned a hardcoded value. Use os.getenv('{var_name}') or os.environ.get('{var_name}') instead.",
                                line=i,
                            ))
                            break  # one issue per line

        # Also detect:  os.environ["VAR"] = "literal"  (hardcoded set)
        m = re.search(r'os\.environ\[[\'"](\w+)[\'"]\]\s*=\s*[\'"]([^\'"]+)[\'"]', line)
        if m:
            var_name = m.group(1)
            value = m.group(2)
            if not re.search(r'\$\{?\w+}?|os\.getenv|getenv|\.env|example|placeholder', value, re.IGNORECASE):
                # Skip if value is a variable ref
                if not re.search(r'os\.environ|environ\.get|getenv', value, re.IGNORECASE):
                    issues.append(SastIssue(
                        "high", "CRED-003",
                        f"Environment variable '{var_name}' hardcoded via os.environ assignment. Read from actual environment instead.",
                        line=i,
                    ))

    return issues


# ── Command injection ───────────────────────────────────────────────────────


def _check_inj_001(content: str) -> list[SastIssue]:
    """INJ-001: os.system() or subprocess with shell=True and string interpolation."""
    issues: list[SastIssue] = []
    lines = content.split("\n")

    for i, line in enumerate(lines, start=1):
        # os.system() with f-string or concatenation
        m = re.search(r'os\.system\s*\(\s*(?:f[\'"]|\'[^\']*\'|\"[^\"]*\"\s*\+)', line)
        if m:
            issues.append(SastIssue(
                "high", "INJ-001",
                "os.system() called with interpolated string — risk of command injection. Use subprocess.run with a list argument instead.",
                line=i,
            ))

        # os.system() with variable + string
        if re.search(r'os\.system\s*\(\s*[\'\"][^\'\"]*[\'\"].*\+|'  # "..." + var
                     r'os\.system\s*\([^\)]*%[\s\(]',  # os.system("... %s" % var)
                     line):
            issues.append(SastIssue(
                "high", "INJ-001",
                "os.system() called with concatenated/format string — risk of command injection.",
                line=i,
            ))

        # subprocess with shell=True and interpolation
        m = re.search(r'subprocess\.[a-zA-Z]+\([^)]*shell=True', line)
        if m:
            # Check for f-strings or concatenation in the command
            if re.search(r'f[\'"]|[\'"]\s*\+|%\s*\(|\.format\(', line):
                issues.append(SastIssue(
                    "high", "INJ-001",
                    "subprocess call with shell=True and interpolated string — risk of command injection. Use shell=False with a list argument.",
                    line=i,
                ))

        # Also check for multi-line: look for shell=True on same or next line
        # with interpolation
        if re.search(r'shell\s*=\s*True', line):
            # Check if previous line had interpolation
            prev_content = "\n".join(lines[:i])  # all content up to this line
            # Simple check: look for an f-string or concat in recent nearby lines
            last_block = "\n".join(lines[max(0, i - 3):i])
            if re.search(rf'f[\'\"][^\'\"]*[\'\"]|[\'\"][^\'\"]*[\'\"]\s*\+', last_block):
                issues.append(SastIssue(
                    "high", "INJ-001",
                    "subprocess call with shell=True and interpolated string nearby — risk of command injection.",
                    line=i,
                ))

    return issues


def _check_inj_002(content: str) -> list[SastIssue]:
    """INJ-002: eval()/exec() with concatenated strings or f-strings."""
    issues: list[SastIssue] = []
    lines = content.split("\n")

    for i, line in enumerate(lines, start=1):
        # eval() with interpolation
        if re.search(r'eval\s*\(\s*f[\'\"][^\'\"]*[\'\"]|'  # eval(f"...")
                     r'eval\s*\([^)]*[\'\"][^\'\"]*[\'\"]\s*\+'  # eval("..." + var)
                     r'|eval\s*\([^)]*\.format\(',  # eval("...".format(...))
                     line):
            issues.append(SastIssue(
                "high", "INJ-002",
                "eval() called with interpolated string — risk of arbitrary code execution.",
                line=i,
            ))

        # exec() with interpolation
        if re.search(r'exec\s*\(\s*f[\'\"][^\'\"]*[\'\"]|'  # exec(f"...")
                     r'exec\s*\([^)]*[\'\"][^\'\"]*[\'\"]\s*\+'  # exec("..." + var)
                     r'|exec\s*\([^)]*\.format\(',  # exec("...".format(...))
                     line):
            issues.append(SastIssue(
                "high", "INJ-002",
                "exec() called with interpolated string — risk of arbitrary code execution.",
                line=i,
            ))

    return issues


def _check_inj_003(content: str) -> list[SastIssue]:
    """INJ-003: os.popen() with interpolation."""
    issues: list[SastIssue] = []
    lines = content.split("\n")

    for i, line in enumerate(lines, start=1):
        if re.search(r'os\.popen\s*\(\s*f[\'\"][^\'\"]*[\'\"]|'  # os.popen(f"...")
                     r'os\.popen\s*\([^)]*[\'\"][^\'\"]*[\'\"]\s*\+'  # os.popen("..." + var)
                     r'|os\.popen\s*\([^)]*%[\s\(]',  # os.popen("... %s" % var)
                     line):
            issues.append(SastIssue(
                "high", "INJ-003",
                "os.popen() called with interpolated string — risk of command injection.",
                line=i,
            ))

    return issues


# ── Unsafe file operations ──────────────────────────────────────────────────


def _check_file_001(content: str) -> list[SastIssue]:
    """FILE-001: shutil.rmtree on a path starting with '/' without validation."""
    issues: list[SastIssue] = []
    lines = content.split("\n")

    for i, line in enumerate(lines, start=1):
        # shutil.rmtree on a path that starts with '/' literal
        m = re.search(r'shutil\.rmtree\s*\(\s*[\'\"](/)', line)
        if m:
            issues.append(SastIssue(
                "high", "FILE-001",
                f"shutil.rmtree() called with absolute path starting with '{m.group(1)}' — risk of destructive deletion without path validation.",
                line=i,
            ))

        # shutil.rmtree with a variable that could be '/'
        m = re.search(r'shutil\.rmtree\s*\(\s*(\w+)', line)
        if m:
            varname = m.group(1)
            # If we can see the variable assigned to '/' nearby, flag it
            for j in range(max(0, i - 5), i):
                prev_line = lines[j]
                if re.search(rf'{varname}\s*=\s*[\'"]/[\'"]', prev_line):
                    issues.append(SastIssue(
                        "high", "FILE-001",
                        f"shutil.rmtree() with variable '{varname}' assigned to root path '/'.",
                        line=i,
                    ))
                    break

    return issues


def _check_file_002(content: str) -> list[SastIssue]:
    """FILE-002: Writing to absolute system paths (/etc, /usr, /boot) without verification."""
    issues: list[SastIssue] = []
    lines = content.split("\n")
    sensitive_dirs = ["/etc", "/usr", "/boot", "/var", "/sys", "/proc", "/bin", "/sbin", "/lib"]

    for i, line in enumerate(lines, start=1):
        for sensitive_dir in sensitive_dirs:
            # Writing to paths like open("/etc/...", "w") or write_text
            if re.search(rf'(?:open|write_text|writelines)\s*\([^)]*[\'\"]({re.escape(sensitive_dir)})', line):
                issues.append(SastIssue(
                    "high", "FILE-002",
                    f"Writing to a system-critical path ('{sensitive_dir}/...') without validation.",
                    line=i,
                ))
                break

    return issues


def _check_file_003(content: str) -> list[SastIssue]:
    """FILE-003: os.remove/unlink on unvalidated paths."""
    issues: list[SastIssue] = []
    lines = content.split("\n")

    for i, line in enumerate(lines, start=1):
        # os.remove with user-supplied input or unvalidated variable
        m = re.search(r'(?:os\.remove|os\.unlink)\s*\(\s*(\w+)', line)
        if m:
            varname = m.group(1)
            # Check if variable looks like user input
            if re.search(r'(user_?input|user_?path|file_?name|arg|sys\.argv|request|form)', varname, re.IGNORECASE):
                issues.append(SastIssue(
                    "medium", "FILE-003",
                    f"os.remove() called with potentially unvalidated path '{varname}'. Validate the path before deletion.",
                    line=i,
                ))

        # os.remove with interpolated path (f-string or concat)
        if re.search(r'(?:os\.remove|os\.unlink)\s*\(\s*f[\'\"][^\'\"]*[\'\"]|'  # f"..."
                     r'(?:os\.remove|os\.unlink)\s*\([^)]*[\'\"][^\'\"]*[\'\"]\s*\+',  # "..." + var
                     line):
            issues.append(SastIssue(
                "medium", "FILE-003",
                "os.remove() called with an interpolated path. Ensure the path is validated before deletion.",
                line=i,
            ))

    return issues


# ── Dangerous imports ────────────────────────────────────────────────────────


def _check_imp_001(content: str) -> list[SastIssue]:
    """IMP-001: Dangerous imports that could lead to security issues."""
    issues: list[SastIssue] = []
    lines = content.split("\n")
    dangerous_imports = {
        "pickle": "Deserializing untrusted data with pickle can lead to arbitrary code execution.",
        "cPickle": "Deserializing untrusted data with cPickle can lead to arbitrary code execution.",
        "ctypes": "ctypes allows raw memory access and can bypass Python's security model.",
        "telnetlib": "Telnet protocol sends data in cleartext (no encryption). Use SSH instead.",
        "ftplib": "FTP sends credentials in cleartext. Use SFTP/FTPS instead.",
        "crypt": "The 'crypt' module uses weak/obsolete password hashing. Use bcrypt or argon2 instead.",
    }

    for i, line in enumerate(lines, start=1):
        for mod_name, warning in dangerous_imports.items():
            if re.search(rf'^\s*import\s+{re.escape(mod_name)}\b', line) or \
               re.search(rf'^\s*from\s+{re.escape(mod_name)}\s+import', line):
                issues.append(SastIssue(
                    "medium", "IMP-001",
                    f"Dangerous import: '{mod_name}'. {warning}",
                    line=i,
                ))

    return issues


# ── Network insecurity ───────────────────────────────────────────────────────


def _check_net_001(content: str) -> list[SastIssue]:
    """NET-001: SSL verification disabled or HTTP URLs for sensitive operations."""
    issues: list[SastIssue] = []
    lines = content.split("\n")

    for i, line in enumerate(lines, start=1):
        # verify=False in requests, httpx, etc.
        if re.search(r'verify\s*=\s*False', line):
            issues.append(SastIssue(
                "high", "NET-001",
                "SSL certificate verification disabled (verify=False). This enables MITM attacks.",
                line=i,
            ))

        # CERT_NONE in ssl contexts
        if re.search(r'cert_reqs\s*=\s*CERT_NONE|ssl\.CERT_NONE', line):
            issues.append(SastIssue(
                "high", "NET-001",
                "SSL certificate requirements set to CERT_NONE — verification disabled.",
                line=i,
            ))

        # ssl._create_default_https_context override
        if re.search(r'ssl\._create_default_https_context\s*=?\s*ssl\._create_unverified_context', line):
            issues.append(SastIssue(
                "high", "NET-001",
                "Global SSL verification disabled by overriding _create_default_https_context.",
                line=i,
            ))

        # HTTP URLs for sensitive operations (API endpoints, auth, webhooks)
        m = re.search(r'http://(?:[a-zA-Z0-9.-]+)(?::\d+)?/(?:api|auth|login|token|webhook|callback|secret)', line)
        if m:
            issues.append(SastIssue(
                "low", "NET-001",
                f"HTTP URL used for sensitive operation: '{m.group(0)}'. Use HTTPS instead.",
                line=i,
            ))

    return issues


# ---------------------------------------------------------------------------
# Main scan function
# ---------------------------------------------------------------------------


def scan_skill(
    content: str,
    skill_dir: Optional[Path] = None,
    skill_name: Optional[str] = None,
) -> list[SastIssue]:
    """Run all SAST checks on skill content.

    Args:
        content: Full SKILL.md content as a string.
        skill_dir: Optional Path to the skill directory (for context).
        skill_name: Optional skill name (for context).

    Returns:
        List of SastIssue findings. Empty list means no issues detected.
    """
    if not content:
        return []

    issues: list[SastIssue] = []

    # 1. Hardcoded credentials
    issues.extend(_check_cred_001(content))
    issues.extend(_check_cred_002(content))
    issues.extend(_check_cred_003(content))

    # 2. Command injection
    issues.extend(_check_inj_001(content))
    issues.extend(_check_inj_002(content))
    issues.extend(_check_inj_003(content))

    # 3. Unsafe file operations
    issues.extend(_check_file_001(content))
    issues.extend(_check_file_002(content))
    issues.extend(_check_file_003(content))

    # 4. Dangerous imports
    issues.extend(_check_imp_001(content))

    # 5. Network insecurity
    issues.extend(_check_net_001(content))

    return issues


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def has_errors(issues: list[SastIssue]) -> bool:
    """Return True if any issue is 'critical' or 'high' severity.

    Critical/high findings block the skill write; medium/low are advisory only.
    """
    return any(issue.severity in ("critical", "high") for issue in issues)


def format_report(issues: list[SastIssue]) -> str:
    """Format a list of SastIssue findings as a human-readable report.

    Groups issues by severity, sorted critical → high → medium → low.
    """
    if not issues:
        return "No security issues found."

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_issues = sorted(issues, key=lambda x: (severity_order.get(x.severity, 99), x.code))

    lines: list[str] = []
    current_severity: Optional[str] = None

    for issue in sorted_issues:
        if issue.severity != current_severity:
            current_severity = issue.severity
            label = current_severity.upper()
            lines.append("")
            lines.append(f"  [{label}]")

        location = ""
        if issue.line is not None:
            location += f":{issue.line}"
        if issue.file:
            location = f" ({issue.file}{location})" if location else f" ({issue.file})"
        elif location:
            location = f" (line {issue.line})"

        lines.append(f"    {issue.code}: {issue.message}{location}")

    lines.append("")
    return "\n".join(lines)