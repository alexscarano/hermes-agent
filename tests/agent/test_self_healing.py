"""Tests for self-healing consecutive-error detection helpers."""

import json
from agent.conversation_loop import _detect_consecutive_error, _find_last_tool_message


def _tool_msg(name: str, content: str) -> dict:
    return {"role": "tool", "name": name, "tool_call_id": "call_1", "content": content}


class TestDetectConsecutiveError:
    def test_terminal_exit_code_error(self):
        msg = _tool_msg("terminal", json.dumps({"exit_code": 1, "error": "No such file: foo.py"}))
        sig = _detect_consecutive_error([msg], None)
        assert sig is not None
        assert sig[0] == "terminal"
        assert "No such file" in sig[1]

    def test_terminal_exit_code_no_error_text(self):
        msg = _tool_msg("terminal", json.dumps({"exit_code": 127}))
        sig = _detect_consecutive_error([msg], None)
        assert sig is not None
        assert sig[0] == "terminal"
        assert "[exit 127]" in sig[1]

    def test_terminal_success(self):
        msg = _tool_msg("terminal", json.dumps({"exit_code": 0, "output": "done"}))
        sig = _detect_consecutive_error([msg], None)
        assert sig is None

    def test_terminal_non_json_error_heuristic(self):
        msg = _tool_msg("terminal", "Error: command not found")
        sig = _detect_consecutive_error([msg], None)
        assert sig is not None
        assert sig[0] == "terminal"

    def test_generic_tool_json_error(self):
        msg = _tool_msg("search_files", json.dumps({"error": "File not found", "success": False}))
        sig = _detect_consecutive_error([msg], None)
        assert sig is not None
        assert sig[0] == "search_files"

    def test_generic_tool_plain_error(self):
        msg = _tool_msg("read_file", "Error: File not found: foo.py")
        sig = _detect_consecutive_error([msg], None)
        assert sig is not None

    def test_success_returns_none(self):
        msg = _tool_msg("web_search", json.dumps({"results": [{"title": "ok"}]}))
        sig = _detect_consecutive_error([msg], None)
        assert sig is None

    def test_empty_content_returns_none(self):
        msg = _tool_msg("terminal", "")
        sig = _detect_consecutive_error([msg], None)
        assert sig is None

    def test_consecutive_same_error(self):
        content = json.dumps({"exit_code": 1, "error": "syntax error"})
        msg = _tool_msg("terminal", content)
        first = _detect_consecutive_error([msg], None)
        second = _detect_consecutive_error([msg], first)
        assert second is not None
        assert second == first

    def test_different_errors_not_matched(self):
        msg1 = _tool_msg("terminal", json.dumps({"exit_code": 1, "error": "syntax error"}))
        msg2 = _tool_msg("terminal", json.dumps({"exit_code": 1, "error": "file not found"}))
        first = _detect_consecutive_error([msg1], None)
        second = _detect_consecutive_error([msg2], first)
        assert second is not None
        assert second != first

    def test_consecutive_detection(self):
        content = json.dumps({"exit_code": 1, "error": "segfault"})
        msg = _tool_msg("run", content)
        sig1 = _detect_consecutive_error([msg], None)
        assert sig1 is not None
        sig2 = _detect_consecutive_error([msg], sig1)
        assert sig2 is not None
        assert sig2 == sig1

    def test_mixed_messages(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
            _tool_msg("terminal", json.dumps({"exit_code": 1, "error": "fail"})),
            {"role": "assistant", "content": "let me fix that"},
        ]
        sig = _detect_consecutive_error(msgs, None)
        assert sig is not None
        assert sig[0] == "terminal"

    def test_multimodal_content(self):
        msg = {
            "role": "tool",
            "name": "vision_analyze",
            "content": [
                {"type": "text", "text": "Error: Could not process image"},
                {"type": "image", "source": {"type": "base64", "data": "..."}},
            ],
        }
        sig = _detect_consecutive_error([msg], None)
        assert sig is not None

    def test_success_after_error_resets(self):
        """A success after an error should return None (new sig, not error)."""
        msgs = [_tool_msg("terminal", json.dumps({"exit_code": 0, "output": "done"}))]
        sig = _detect_consecutive_error(msgs, ("terminal", "error snippet"))
        assert sig is None

    def test_non_tool_message_skipped(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
        sig = _detect_consecutive_error(msgs, None)
        assert sig is None


class TestFindLastToolMessage:
    def test_finds_last_tool_message(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "name": "terminal", "content": "out1"},
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "name": "memory", "content": "out2"},
        ]
        last = _find_last_tool_message(msgs)
        assert last is not None
        assert last["name"] == "memory"

    def test_no_tool_message(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert _find_last_tool_message(msgs) is None

    def test_empty_list(self):
        assert _find_last_tool_message([]) is None
