"""AST-based static analysis for Python code and enhanced pattern-based shell code checking.

This module catches dangerous patterns that the string-based DANGEROUS_PATTERNS in
approval.py misses — particularly AST-level constructs (os.system, subprocess with
shell=True, eval/exec with non-literal args, __import__ injection, os.environ
overwrites, builtin shadowing, dangerous open() calls, and chmod with 777/666)
and shell-level patterns that survive obfuscation, heredocs, and multi-line encoding.

Usage:
    from tools.code_guardrails import check_code, has_critical_or_high, format_guardrail_report

    issues = check_code("import os; os.system('rm -rf /')", language="python")
    if has_critical_or_high(issues):
        print(format_guardrail_report(issues))
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# GuardrailIssue
# ---------------------------------------------------------------------------

@dataclass
class GuardrailIssue:
    """A single static-analysis finding against Python or shell code."""

    severity: str  # "critical" | "high" | "medium" | "low"
    code: str      # ex: "AST-001"
    message: str
    line: Optional[int] = None


# ---------------------------------------------------------------------------
# Overridable defaults — kept as module-level lists so callers can inspect or
# extend them without monkeypatching.  Each list element is:
#   (rule_code, severity, message_template, check_fn)
# ---------------------------------------------------------------------------

_PYTHON_RULES: list[tuple[str, str, str, callable]] = []
_SHELL_RULES: list[tuple[str, str, str, callable]] = []


def _register_python(rule_code: str, severity: str, message: str, check_fn: callable) -> None:
    _PYTHON_RULES.append((rule_code, severity, message, check_fn))


def _register_shell(rule_code: str, severity: str, message: str, check_fn: callable) -> None:
    _SHELL_RULES.append((rule_code, severity, message, check_fn))


# ===================================================================
# Python AST rule implementations
# ===================================================================

# --- AST-001: subprocess.* / os.system / os.popen with shell=True ---

class _ShellTrueVisitor(ast.NodeVisitor):
    """Detect subprocess.*(..., shell=True) and os.system/popen calls."""

    def __init__(self) -> None:
        self.issues: list[GuardrailIssue] = []

    def _check_call(self, node: ast.Call, rule_code: str, severity: str, message: str) -> None:
        """Check a single Call node for shell=True keyword."""
        for kw in node.keywords:
            if kw.arg == "shell" and _is_literal_true(kw.value):
                self.issues.append(GuardrailIssue(
                    severity=severity,
                    code=rule_code,
                    message=message,
                    line=node.lineno,
                ))
                return  # one issue per call

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        # subprocess.call/check_call/check_output/Popen/run(..., shell=True)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "subprocess" and func.attr in {
                "call", "check_call", "check_output", "Popen", "run",
            }:
                self._check_call(node, "AST-001", "critical",
                                 "subprocess.{0}() called with shell=True — "
                                 "bypasses string-based approval".format(func.attr))
        # os.system(...) or os.popen(...)  (shell=True implicit for os.system)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "os" and func.attr in ("system", "popen"):
                self.issues.append(GuardrailIssue(
                    severity="critical",
                    code="AST-001",
                    message=f"os.{func.attr}() called — executes shell command via subprocess with implicit shell",
                    line=node.lineno,
                ))
        self.generic_visit(node)


def _check_ast_001(tree: ast.AST) -> list[GuardrailIssue]:
    visitor = _ShellTrueVisitor()
    visitor.visit(tree)
    return visitor.issues


_register_python(
    "AST-001",
    "critical",
    "subprocess/os.system/os.popen with implicit or explicit shell",
    _check_ast_001,
)


# --- AST-002: eval() / exec() with non-literal argument ---

class _EvalExecVisitor(ast.NodeVisitor):
    """Detect eval(expr) or exec(code) where the argument is a variable, not a literal."""

    def __init__(self) -> None:
        self.issues: list[GuardrailIssue] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in {"eval", "exec"}:
            if node.args:
                arg = node.args[0]
                # Allow string literals (constant strings, f-strings without
                # interpolation, or joined string literals)
                if not _is_safe_literal_expr(arg):
                    self.issues.append(GuardrailIssue(
                        severity="critical",
                        code="AST-002",
                        message=f"{func.id}() called with non-literal argument — possible code injection",
                        line=node.lineno,
                    ))
        self.generic_visit(node)


def _check_ast_002(tree: ast.AST) -> list[GuardrailIssue]:
    visitor = _EvalExecVisitor()
    visitor.visit(tree)
    return visitor.issues


_register_python(
    "AST-002",
    "critical",
    "eval/exec with non-literal argument",
    _check_ast_002,
)


# --- AST-003: __import__() dynamic with variable module name ---

class _DynamicImportVisitor(ast.NodeVisitor):
    """Detect __import__(var) where module name comes from a variable."""

    def __init__(self) -> None:
        self.issues: list[GuardrailIssue] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id == "__import__":
            if node.args:
                arg = node.args[0]
                if not _is_safe_literal_expr(arg):
                    self.issues.append(GuardrailIssue(
                        severity="high",
                        code="AST-003",
                        message="__import__() called with dynamic module name — possible module injection",
                        line=node.lineno,
                    ))
        # Also catch importlib.import_module(var)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "importlib" and func.attr == "import_module":
                if node.args:
                    arg = node.args[0]
                    if not _is_safe_literal_expr(arg):
                        self.issues.append(GuardrailIssue(
                            severity="high",
                            code="AST-003",
                            message="importlib.import_module() called with dynamic module name",
                            line=node.lineno,
                        ))
        self.generic_visit(node)


def _check_ast_003(tree: ast.AST) -> list[GuardrailIssue]:
    visitor = _DynamicImportVisitor()
    visitor.visit(tree)
    return visitor.issues


_register_python(
    "AST-003",
    "high",
    "Dynamic __import__() or importlib.import_module()",
    _check_ast_003,
)


# --- AST-004: Assignment to os.environ with restricted variable ---

_RESTRICTED_ENV_VARS = frozenset({
    "PATH", "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH",
    "HOME", "SHELL", "IFS", "BASH_ENV", "ENV",
})


class _EnvironAssignVisitor(ast.NodeVisitor):
    """Detect os.environ['VAR'] = val for restricted variables."""

    def __init__(self) -> None:
        self.issues: list[GuardrailIssue] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            # os.environ['VAR'] = ...
            if isinstance(target, ast.Subscript):
                subj = target.value
                if isinstance(subj, ast.Attribute) and isinstance(subj.value, ast.Name):
                    if subj.value.id == "os" and subj.attr == "environ":
                        key = target.slice
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            if key.value in _RESTRICTED_ENV_VARS:
                                self.issues.append(GuardrailIssue(
                                    severity="high",
                                    code="AST-004",
                                    message=f"os.environ['{key.value}'] assignment — restricted environment variable",
                                    line=node.lineno,
                                ))
                        elif isinstance(key, ast.Name):
                            # os.environ[var] where var is a runtime variable
                            self.issues.append(GuardrailIssue(
                                severity="high",
                                code="AST-004",
                                message=f"os.environ[{key.id}] assignment — dynamic restricted environment variable",
                                line=node.lineno,
                            ))
        # Also catch os.environ.update({'VAR': ...})
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
            if isinstance(func.value.value, ast.Name):
                if func.value.value.id == "os" and func.value.attr == "environ" and func.attr == "update":
                    for arg in node.args:
                        if isinstance(arg, (ast.Dict, ast.Call)):
                            self.issues.append(GuardrailIssue(
                                severity="high",
                                code="AST-004",
                                message="os.environ.update() — potential restricted environment variable overwrite",
                                line=node.lineno,
                            ))
        self.generic_visit(node)


def _check_ast_004(tree: ast.AST) -> list[GuardrailIssue]:
    visitor = _EnvironAssignVisitor()
    visitor.visit(tree)
    return visitor.issues


_register_python(
    "AST-004",
    "high",
    "Assignment to restricted environment variable",
    _check_ast_004,
)


# --- AST-005: shutil.rmtree() without path validation ---

class _RmtreeVisitor(ast.NodeVisitor):
    """Detect shutil.rmtree() with potentially dangerous paths."""

    def __init__(self) -> None:
        self.issues: list[GuardrailIssue] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "shutil" and func.attr == "rmtree":
                if node.args:
                    path_arg = node.args[0]
                    # Flag if the path is a variable (not a literal string)
                    if not _is_safe_literal_expr(path_arg):
                        self.issues.append(GuardrailIssue(
                            severity="critical",
                            code="AST-005",
                            message="shutil.rmtree() called with non-literal path — possible destructive deletion",
                            line=node.lineno,
                        ))
                    elif isinstance(path_arg, ast.Constant) and isinstance(path_arg.value, str):
                        path_val = path_arg.value
                        if path_val in ("/", "//", "/.", "/..") or path_val.startswith("/") and _is_root_like(path_val):
                            self.issues.append(GuardrailIssue(
                                severity="critical",
                                code="AST-005",
                                message=f"shutil.rmtree() called with literal path '{path_val}' — risk of system deletion",
                                line=node.lineno,
                            ))
        self.generic_visit(node)


def _is_root_like(path: str) -> bool:
    """Check if a path resolves to something dangerous like root."""
    # Normalize
    normalized = re.sub(r'/+', '/', path)
    if normalized in ("/", "//", "/.", "/..", ""):
        return True
    # Check for /*, /.* etc
    if re.match(r'^/(\*|\.\*|)$', normalized):
        return True
    return False


def _check_ast_005(tree: ast.AST) -> list[GuardrailIssue]:
    visitor = _RmtreeVisitor()
    visitor.visit(tree)
    return visitor.issues


_register_python(
    "AST-005",
    "critical",
    "shutil.rmtree() with unvalidated path",
    _check_ast_005,
)


# --- AST-006: Builtin function overwrite ---

_BUILTIN_FUNCS = frozenset({
    "open", "print", "eval", "exec", "input", "import",
    "execfile", "compile", "__import__", "getattr", "setattr",
    "delattr", "locals", "globals", "vars", "dir",
})


class _BuiltinOverwriteVisitor(ast.NodeVisitor):
    """Detect overwriting of built-in functions."""

    def __init__(self) -> None:
        self.issues: list[GuardrailIssue] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in _BUILTIN_FUNCS:
                self.issues.append(GuardrailIssue(
                    severity="high",
                    code="AST-006",
                    message=f"Built-in '{target.id}' is being overwritten — possible shadowing attack",
                    line=node.lineno,
                ))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name in _BUILTIN_FUNCS:
            self.issues.append(GuardrailIssue(
                severity="high",
                code="AST-006",
                message=f"Built-in '{node.name}' is being redefined as a function — possible shadowing attack",
                line=node.lineno,
            ))
        self.generic_visit(node)


def _check_ast_006(tree: ast.AST) -> list[GuardrailIssue]:
    visitor = _BuiltinOverwriteVisitor()
    visitor.visit(tree)
    return visitor.issues


_register_python(
    "AST-006",
    "high",
    "Overwriting a built-in function",
    _check_ast_006,
)


# --- AST-007: open() with write mode to system paths ---

_SYSTEM_WRITE_PATHS = (
    "/etc/", "/usr/", "/boot/", "/dev/",
)


class _SystemWriteVisitor(ast.NodeVisitor):
    """Detect open(path, 'w'/'wb') where path is a system absolute path."""

    def __init__(self) -> None:
        self.issues: list[GuardrailIssue] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        # open(...) built-in call
        if isinstance(func, ast.Name) and func.id == "open":
            mode = "r"  # default
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            # Also check positional second arg
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)

            if mode in ("w", "wb", "x", "xb", "a"):
                # Check path (first positional arg)
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    path = node.args[0].value
                    for prefix in _SYSTEM_WRITE_PATHS:
                        if path.startswith(prefix):
                            self.issues.append(GuardrailIssue(
                                severity="high",
                                code="AST-007",
                                message=f"open() with mode '{mode}' on system path '{prefix}...' — possible system file overwrite",
                                line=node.lineno,
                            ))
                            break
        self.generic_visit(node)


def _check_ast_007(tree: ast.AST) -> list[GuardrailIssue]:
    visitor = _SystemWriteVisitor()
    visitor.visit(tree)
    return visitor.issues


_register_python(
    "AST-007",
    "high",
    "open() with write mode to system path",
    _check_ast_007,
)


# --- AST-008: os.chmod() with 777 or 666 permissions ---

class _ChmodVisitor(ast.NodeVisitor):
    """Detect os.chmod(path, 0o777) or os.chmod(path, 0o666)."""

    def __init__(self) -> None:
        self.issues: list[GuardrailIssue] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "os" and func.attr == "chmod":
                if len(node.args) >= 2:
                    perm = node.args[1]
                    # Check for 0o777 or 0o666 literal
                    if isinstance(perm, ast.Constant):
                        val = perm.value
                        if val == 0o777 or val == 511:
                            self.issues.append(GuardrailIssue(
                                severity="high",
                                code="AST-008",
                                message="os.chmod() with 0o777 permissions — world-writable",
                                line=node.lineno,
                            ))
                        elif val == 0o666 or val == 438:
                            self.issues.append(GuardrailIssue(
                                severity="high",
                                code="AST-008",
                                message="os.chmod() with 0o666 permissions — world-writable",
                                line=node.lineno,
                            ))
        self.generic_visit(node)


def _check_ast_008(tree: ast.AST) -> list[GuardrailIssue]:
    visitor = _ChmodVisitor()
    visitor.visit(tree)
    return visitor.issues


_register_python(
    "AST-008",
    "high",
    "os.chmod() with 777/666 permissions",
    _check_ast_008,
)


# ===================================================================
# Shell rule implementations
# ===================================================================

# --- SHS-001: rm -rf / or /* in any form ---

def _check_shs_001(code: str) -> list[GuardrailIssue]:
    """Detect rm -rf /, /*, /. etc in any shell context.

    Uses end-of-pattern anchors (lookahead for whitespace, end-of-string, or
    command separators) instead of ``\\b`` because ``/``, ``*``, and ``.`` are
    non-word characters where ``\\b`` never fires.
    """
    issues: list[GuardrailIssue] = []
    # End-of-path anchor: whitespace, EOS, pipe, semicolon, ampersand,
    # closing quote, closing paren/brace/backtick, or redirection operator.
    _EOP = r"""(?=\s|$|[|;&)`}'"<>])"""
    # Root or root-dot anchored paths — but not /tmp/, /var/, /home/, /dev/,
    # etc. which are legitimate clean-up targets.
    _SAFE_TOP_DIRS = r'(?:tmp|var|home|dev)\b'
    _ROOT_LIKE = rf'/(?:$|\*|\.\*?|\.\.\*?|(?!{_SAFE_TOP_DIRS})[^/\s]+)' + _EOP
    # Quoted form: rm -rf "/" or rm -rf '/*'
    _ROOT_QUOTED = r'["\']/' + _EOP
    # HOME variable expansion
    _HOME_PAT = r'\$\{?HOME\}?' + _EOP

    patterns = [
        rf'\brm\s+(?:-[^\s]*\s+)*{_ROOT_LIKE}',
        rf'\brm\s+(?:-[^\s]*\s+)*{_ROOT_QUOTED}',
        rf'\brm\s+(?:-[^\s]*\s+)*{_HOME_PAT}',
    ]
    for pat in patterns:
        for match in re.finditer(pat, code, re.IGNORECASE):
            line = _line_from_match(code, match)
            if _is_comment(code, match.start()):
                continue
            matched = match.group().strip()
            issues.append(GuardrailIssue(
                severity="critical",
                code="SHS-001",
                message=f"Destructive rm command targeting root filesystem: '{matched}'",
                line=line,
            ))
    return issues


# --- SHS-002: base64 | bash in heredocs ---

def _check_shs_002(code: str) -> list[GuardrailIssue]:
    """Detect base64 decode piped to shell, including inside heredocs."""
    issues: list[GuardrailIssue] = []
    pat = r'\b(base64|base32)\s+(?:-[dD]|--decode)\b[^;|\n]*\|[^;|\n]*\b(bash|sh|zsh|ksh)\b'
    for match in re.finditer(pat, code, re.IGNORECASE):
        issues.append(GuardrailIssue(
            severity="critical",
            code="SHS-002",
            message=f"Decode-and-execute pipeline: '{match.group().strip()}'",
            line=_line_from_match(code, match),
        ))
    return issues


# --- SHS-003: curl/wget | bash in heredocs or strings ---

def _check_shs_003(code: str) -> list[GuardrailIssue]:
    """Detect curl/wget piped to shell, including inside heredocs or Python strings."""
    issues: list[GuardrailIssue] = []
    pat = r'\b(curl|wget)\b[^;|\n]*\|[^;|\n]*\b(bash|sh|zsh|ksh)\b'
    for match in re.finditer(pat, code, re.IGNORECASE):
        issues.append(GuardrailIssue(
            severity="high",
            code="SHS-003",
            message=f"Pipe remote content to shell: '{match.group().strip()}'",
            line=_line_from_match(code, match),
        ))
    return issues


# --- SHS-004: Dangerous commands inside heredocs ---

def _check_shs_004(code: str) -> list[GuardrailIssue]:
    """Detect dangerous patterns inside heredocs (<<EOF ... EOF)."""
    issues: list[GuardrailIssue] = []
    # Match heredoc boundaries
    heredoc_pat = re.compile(
        r'(?:<<\s*(\w+)\s*\n)(.*?)\n\1\b',
        re.DOTALL,
    )
    for hdoc_match in heredoc_pat.finditer(code):
        hdoc_body = hdoc_match.group(2)
        # Look for dangerous commands inside the heredoc
        danger_signals = [
            (r'\brm\s+-rf\b', "rm -rf inside heredoc"),
            (r'\b(base64|base32)\b.*\|.*\b(bash|sh)\b', "decode-and-execute inside heredoc"),
            (r'\b(curl|wget)\b.*\|.*\b(bash|sh)\b', "pipe remote content inside heredoc"),
            (r'\bdpkg\b.*--purge\b', "package purge inside heredoc"),
            (r'\bchmod\s+777\b', "world-writable permissions inside heredoc"),
        ]
        for danger_pat, desc in danger_signals:
            inner_match = re.search(danger_pat, hdoc_body, re.IGNORECASE)
            if inner_match:
                issues.append(GuardrailIssue(
                    severity="high",
                    code="SHS-004",
                    message=f"Dangerous command found inside heredoc: {desc}",
                    line=_line_from_match(code, hdoc_match),
                ))
                break  # one issue per heredoc
    return issues


# --- SHS-005: Redirection to /dev/ or block devices ---

def _check_shs_005(code: str) -> list[GuardrailIssue]:
    """Detect redirection to /dev/ paths (not /dev/null) or block devices."""
    issues: list[GuardrailIssue] = []
    # Skip /dev/null explicitly
    pat = r'(?:>|>>|1>|2>)\s*/dev/(?!null\b)[a-z0-9]+(?:\s|$|;|&|\|)'
    for match in re.finditer(pat, code, re.IGNORECASE):
        issues.append(GuardrailIssue(
            severity="high",
            code="SHS-005",
            message=f"Redirection to /dev/ device (non-null): '{match.group().strip()}'",
            line=_line_from_match(code, match),
        ))
    return issues


_register_shell(
    "SHS-001",
    "critical",
    "rm -rf targeting root filesystem",
    _check_shs_001,
)
_register_shell(
    "SHS-002",
    "critical",
    "Decode-and-execute pipeline",
    _check_shs_002,
)
_register_shell(
    "SHS-003",
    "high",
    "Pipe remote content to shell",
    _check_shs_003,
)
_register_shell(
    "SHS-004",
    "high",
    "Dangerous commands inside heredoc",
    _check_shs_004,
)
_register_shell(
    "SHS-005",
    "high",
    "Redirection to /dev/ block device",
    _check_shs_005,
)


# ===================================================================
# Helper functions
# ===================================================================

def _is_literal_true(node: ast.AST) -> bool:
    """Check if an AST node is a literal True (constant or Name)."""
    if isinstance(node, ast.Constant) and node.value is True:
        return True
    if isinstance(node, ast.Name) and node.id == "True":
        return True
    return False


def _is_safe_literal_expr(node: ast.AST) -> bool:
    """Check if an expression is a safe literal (constant string, JoinedStr of constants, etc.)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.JoinedStr):
        # All values must be constants
        return all(isinstance(v, ast.Constant) and isinstance(v.value, str) for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        return _is_safe_literal_expr(node.left) and _is_safe_literal_expr(node.right)
    return False


def _line_from_match(code: str, match: re.Match) -> int:
    """Return the 1-based line number of a regex match."""
    return code[:match.start()].count("\n") + 1


def _is_comment(code: str, pos: int) -> bool:
    """Check if the position is inside a shell comment (#)."""
    line_start = code.rfind("\n", 0, pos) + 1 if "\n" in code[:pos] else 0
    before = code[line_start:pos]
    # Simple heuristic: if there's a # before the match on the same line
    # and it's not inside quotes
    return "#" in before


# ===================================================================
# Public API
# ===================================================================

def check_python_code(code: str) -> list[GuardrailIssue]:
    """Run AST-based checks on Python code.

    Parses the code into an AST and runs all registered Python rules.
    Returns a list of GuardrailIssue findings (empty list if safe).
    """
    if not code or not code.strip():
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        # Return a single issue for unparseable code (still better than
        # allowing it to run unchecked)
        return [GuardrailIssue(
            severity="medium",
            code="AST-PARSE",
            message=f"Code failed to parse: {exc}",
            line=exc.lineno,
        )]

    issues: list[GuardrailIssue] = []
    for _code, _severity, _msg, check_fn in _PYTHON_RULES:
        try:
            issues.extend(check_fn(tree))
        except Exception:
            # Individual rule failures should not silence other rules
            pass
    return issues


def check_shell_code(code: str) -> list[GuardrailIssue]:
    """Run enhanced shell pattern checks on shell code.

    Goes beyond DANGEROUS_PATTERNS from approval.py: detects obfuscation,
    multi-line code, heredoc-dwelling dangerous commands.
    """
    if not code or not code.strip():
        return []

    issues: list[GuardrailIssue] = []
    for _code, _severity, _msg, check_fn in _SHELL_RULES:
        try:
            issues.extend(check_fn(code))
        except Exception:
            pass
    return issues


def check_code(code: str, language: str = "python") -> list[GuardrailIssue]:
    """Unified dispatcher that runs the appropriate checks.

    Args:
        code: The source code to analyze.
        language: One of "python", "shell", or "auto" (default: "python").
                  "auto" detects python vs shell by shebang or syntax.

    Returns:
        List of GuardrailIssue findings.
    """
    lang = language.lower()
    if lang == "auto":
        lang = _detect_language(code)
    if lang == "python":
        return check_python_code(code)
    elif lang in ("shell", "bash", "sh", "zsh"):
        return check_shell_code(code)
    else:
        return [GuardrailIssue(
            severity="low",
            code="UNKNOWN-LANG",
            message=f"Unknown language: '{language}'. Supported: python, shell.",
            line=None,
        )]


def _detect_language(code: str) -> str:
    """Detect if code is Python or shell based on content."""
    stripped = code.strip()
    if not stripped:
        return "python"
    # Shebang detection
    if stripped.startswith("#!/") or stripped.startswith("#!"):
        if "python" in stripped.split("\n")[0].lower():
            return "python"
        return "shell"
    # Try parsing as Python
    try:
        ast.parse(code)
        return "python"
    except SyntaxError:
        return "shell"


def has_critical_or_high(issues: list[GuardrailIssue]) -> bool:
    """Return True if any issue has severity 'critical' or 'high'."""
    return any(i.severity in ("critical", "high") for i in issues)


def format_guardrail_report(issues: list[GuardrailIssue]) -> str:
    """Format a list of GuardrailIssue into a human-readable report."""
    if not issues:
        return "No issues found."

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("CODE GUARDRAIL REPORT")
    lines.append("=" * 60)

    # Sort: critical first, then high, etc.
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_issues = sorted(issues, key=lambda i: (severity_order.get(i.severity, 99), i.code))

    for issue in sorted_issues:
        loc = f" (line {issue.line})" if issue.line is not None else ""
        tag = f"[{issue.severity.upper()}]"
        lines.append(f"  {tag} {issue.code}: {issue.message}{loc}")

    lines.append("-" * 60)
    critical = sum(1 for i in issues if i.severity == "critical")
    high = sum(1 for i in issues if i.severity == "high")
    total = len(issues)
    lines.append(f"Summary: {total} issue(s) — {critical} critical, {high} high")
    lines.append("=" * 60)
    return "\n".join(lines)