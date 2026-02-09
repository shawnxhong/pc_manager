import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure the project root is on sys.path so imports work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub out heavy native dependencies that may not be installed in test environments,
# so we can still test the pure-Python utility functions from run_agent.
for mod_name in [
    "openvino", "openvino.properties", "openvino.properties.hint",
    "openvino.properties.streams", "openvino_genai",
    "tools", "pc_manager_prompt",
]:
    sys.modules.setdefault(mod_name, MagicMock())

try:
    from run_agent import _parse_tool_request, _sanitize_pc_manager_args, MAX_TOOL_CALLS, LangGraphAgentRunner
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"Skipping tests, missing dependency: {exc}") from exc


class ParseToolRequestTests(unittest.TestCase):
    """Tests for _parse_tool_request edge cases."""

    def test_valid_action_json(self):
        text = 'Action: realtime_weather\nAction Input: {"city": "shanghai"}'
        result = _parse_tool_request(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "realtime_weather")
        self.assertEqual(result["args"], {"city": "shanghai"})

    def test_trailing_period(self):
        text = 'Action: realtime_weather\nAction Input: {"city": "berlin"}.'
        result = _parse_tool_request(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["args"], {"city": "berlin"})

    def test_no_action(self):
        text = "I think the weather is nice today."
        result = _parse_tool_request(text)
        self.assertIsNone(result)

    def test_empty_string(self):
        result = _parse_tool_request("")
        self.assertIsNone(result)

    def test_malformed_json_fallback(self):
        text = "Action: some_tool\nAction Input: not json at all"
        result = _parse_tool_request(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "some_tool")
        self.assertEqual(result["args"], {"input": "not json at all"})

    def test_json_embedded_in_text(self):
        text = 'Action: pc_manager_search\nAction Input: Here is the input {"intent": "display"} done'
        result = _parse_tool_request(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["args"], {"intent": "display"})

    def test_code_fenced_json(self):
        text = 'Action: pc_manager_search\nAction Input: ```{"intent": "wifi"}```'
        result = _parse_tool_request(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["args"], {"intent": "wifi"})

    def test_action_with_hyphen(self):
        text = 'Action: my-tool\nAction Input: {"key": "val"}'
        result = _parse_tool_request(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "my-tool")

    def test_extra_whitespace(self):
        text = 'Action :  realtime_weather  \n  Action Input :  {"city": "tokyo"}'
        result = _parse_tool_request(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "realtime_weather")
        self.assertEqual(result["args"], {"city": "tokyo"})


class SanitizePcManagerArgsTests(unittest.TestCase):
    """Tests for _sanitize_pc_manager_args."""

    def test_non_pc_tool_passthrough(self):
        args = {"city": "paris"}
        result = _sanitize_pc_manager_args("realtime_weather", args)
        self.assertEqual(result, {"city": "paris"})

    def test_input_to_intent_promotion(self):
        args = {"input": "open wifi"}
        result = _sanitize_pc_manager_args("pc_manager_search", args)
        self.assertEqual(result["intent"], "open wifi")

    def test_dict_intent_extraction(self):
        args = {"intent": {"title": "Display", "value": "ms-settings:display"}}
        result = _sanitize_pc_manager_args("pc_manager_open", args)
        self.assertEqual(result["intent"], "Display")

    def test_non_string_intent_coerced(self):
        args = {"intent": 123}
        result = _sanitize_pc_manager_args("pc_manager_search", args)
        self.assertIsInstance(result["intent"], str)
        self.assertEqual(result["intent"], "123")

    def test_non_dict_args_returns_empty(self):
        result = _sanitize_pc_manager_args("pc_manager_search", "bad")
        self.assertEqual(result, {})


class MaxToolCallGuardTests(unittest.TestCase):
    """Tests for the MAX_TOOL_CALLS agent loop guard."""

    def test_max_tool_calls_constant_exists(self):
        self.assertIsInstance(MAX_TOOL_CALLS, int)
        self.assertGreater(MAX_TOOL_CALLS, 0)

    def test_agent_stops_after_max_tool_calls(self):
        """Agent should stop issuing tool calls after hitting the limit."""
        class AlwaysCallToolLLM:
            def chat(self, messages, stream=False, generate_cfg=None):
                return {"content": 'Action: fake_tool\nAction Input: {"x": 1}'}

        class FakeTool:
            name = "fake_tool"
            description = "A fake tool"
            args = {"x": "int"}

            def __init__(self):
                self.call_count = 0

            def invoke(self, args):
                self.call_count += 1
                return '{"ok": true}'

        tool = FakeTool()
        runner = LangGraphAgentRunner(
            llm=AlwaysCallToolLLM(),
            tools=[tool],
            system_prompt="Test",
        )
        messages = [{"role": "user", "content": "do something"}]
        output = runner.invoke(messages, recursion_limit=MAX_TOOL_CALLS * 4)

        # The tool should have been called at most MAX_TOOL_CALLS times
        self.assertLessEqual(tool.call_count, MAX_TOOL_CALLS)
        # The final message should be from the assistant explaining the limit
        self.assertEqual(output[-1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
