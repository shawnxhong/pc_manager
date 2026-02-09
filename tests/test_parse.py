import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from run_agent import (
        _parse_tool_request,
        _likely_pc_action,
        _sanitize_pc_manager_args,
        _extract_text,
        _messages_to_prompt,
    )
    _HAS_RUN_AGENT = True
except ImportError:
    _HAS_RUN_AGENT = False


@unittest.skipUnless(_HAS_RUN_AGENT, "run_agent deps not available")
class TestParseToolRequest(unittest.TestCase):
    def test_basic_action(self):
        text = 'Action: realtime_weather\nAction Input: {"city": "shanghai"}'
        result = _parse_tool_request(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "realtime_weather")
        self.assertEqual(result["args"], {"city": "shanghai"})

    def test_trailing_period(self):
        text = 'Action: realtime_weather\nAction Input: {"city": "shanghai"}.'
        result = _parse_tool_request(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["args"], {"city": "shanghai"})

    def test_no_action(self):
        text = "The weather in Shanghai is sunny."
        result = _parse_tool_request(text)
        self.assertIsNone(result)

    def test_malformed_json_fallback(self):
        text = "Action: pc_manager_open\nAction Input: open settings"
        result = _parse_tool_request(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "pc_manager_open")
        self.assertEqual(result["args"], {"input": "open settings"})

    def test_code_block_stripped(self):
        text = 'Action: realtime_weather\nAction Input: ```{"city": "tokyo"}```'
        result = _parse_tool_request(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["args"], {"city": "tokyo"})

    def test_embedded_json_extraction(self):
        text = 'Action: pc_manager_open\nAction Input: Please open {"intent": "display"} now'
        result = _parse_tool_request(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["args"], {"intent": "display"})

    def test_hyphenated_tool_name(self):
        text = 'Action: pc-manager-open\nAction Input: {"intent": "display"}'
        result = _parse_tool_request(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "pc-manager-open")


@unittest.skipUnless(_HAS_RUN_AGENT, "run_agent deps not available")
class TestLikelyPcAction(unittest.TestCase):
    def test_open_keyword(self):
        self.assertTrue(_likely_pc_action("open task manager"))

    def test_settings_keyword(self):
        self.assertTrue(_likely_pc_action("go to settings"))

    def test_display_keyword(self):
        self.assertTrue(_likely_pc_action("my display is broken"))

    def test_volume_keyword(self):
        self.assertTrue(_likely_pc_action("change volume"))

    def test_weather_excluded(self):
        self.assertFalse(_likely_pc_action("what's the weather"))

    def test_wikipedia_excluded(self):
        self.assertFalse(_likely_pc_action("search wikipedia for cats"))

    def test_chinese_open(self):
        self.assertTrue(_likely_pc_action("打开设置"))

    def test_chinese_bluetooth(self):
        self.assertTrue(_likely_pc_action("蓝牙"))

    def test_empty_string(self):
        self.assertFalse(_likely_pc_action(""))

    def test_none(self):
        self.assertFalse(_likely_pc_action(None))

    def test_unrelated(self):
        self.assertFalse(_likely_pc_action("hello how are you"))

    def test_case_insensitive(self):
        self.assertTrue(_likely_pc_action("OPEN TASK MANAGER"))


@unittest.skipUnless(_HAS_RUN_AGENT, "run_agent deps not available")
class TestSanitizePcManagerArgs(unittest.TestCase):
    def test_non_pc_tool_passthrough(self):
        args = {"city": "shanghai"}
        result = _sanitize_pc_manager_args("realtime_weather", args)
        self.assertEqual(result, {"city": "shanghai"})

    def test_input_to_intent(self):
        args = {"input": "open settings"}
        result = _sanitize_pc_manager_args("pc_manager_open", args)
        self.assertEqual(result["intent"], "open settings")

    def test_dict_intent_title(self):
        args = {"intent": {"title": "display settings"}}
        result = _sanitize_pc_manager_args("pc_manager_open", args)
        self.assertEqual(result["intent"], "display settings")

    def test_dict_intent_value(self):
        args = {"intent": {"value": "screen brightness"}}
        result = _sanitize_pc_manager_args("pc_manager_open", args)
        self.assertEqual(result["intent"], "screen brightness")

    def test_non_dict_args_returns_empty(self):
        result = _sanitize_pc_manager_args("pc_manager_open", "not a dict")
        self.assertEqual(result, {})

    def test_search_tool_also_sanitized(self):
        args = {"input": "wifi"}
        result = _sanitize_pc_manager_args("pc_manager_search", args)
        self.assertEqual(result["intent"], "wifi")

    def test_non_string_intent(self):
        args = {"intent": 123}
        result = _sanitize_pc_manager_args("pc_manager_open", args)
        self.assertEqual(result["intent"], "123")


@unittest.skipUnless(_HAS_RUN_AGENT, "run_agent deps not available")
class TestExtractText(unittest.TestCase):
    def test_string_input(self):
        self.assertEqual(_extract_text("hello"), "hello")

    def test_none_input(self):
        self.assertEqual(_extract_text(None), "")

    def test_dict_with_content(self):
        self.assertEqual(_extract_text({"content": "text"}), "text")

    def test_dict_with_list_content(self):
        result = _extract_text({"content": ["a", "b"]})
        self.assertEqual(result, "a b")

    def test_list_of_messages(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _extract_text(msgs)
        self.assertEqual(result, "hello")


@unittest.skipUnless(_HAS_RUN_AGENT, "run_agent deps not available")
class TestMessagesToPrompt(unittest.TestCase):
    def test_basic_format(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = _messages_to_prompt(msgs)
        self.assertIn("<|im_start|>system", result)
        self.assertIn("<|im_start|>user", result)
        self.assertIn("<|im_start|>assistant", result)
        self.assertIn("<|im_end|>", result)

    def test_tool_role_mapped_to_observation(self):
        msgs = [
            {"role": "tool", "name": "weather", "content": '{"temp": 25}'},
        ]
        result = _messages_to_prompt(msgs)
        self.assertIn("<|im_start|>observation", result)
        self.assertIn("[weather]", result)

    def test_function_role_mapped_to_observation(self):
        msgs = [
            {"role": "function", "name": "search", "content": "result"},
        ]
        result = _messages_to_prompt(msgs)
        self.assertIn("<|im_start|>observation", result)
        self.assertIn("[search]", result)

    def test_ends_with_assistant_start(self):
        msgs = [{"role": "user", "content": "test"}]
        result = _messages_to_prompt(msgs)
        self.assertTrue(result.strip().endswith("<|im_start|>assistant"))


if __name__ == "__main__":
    unittest.main()
