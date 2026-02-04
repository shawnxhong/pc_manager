import unittest

try:
    from run_agent import LangGraphAgentRunner
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"Skipping tests, missing dependency: {exc}") from exc


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages, stream=False):
        self.calls += 1
        if not self._responses:
            raise AssertionError("No more fake responses configured.")
        return {"content": self._responses.pop(0)}


class FakeTool:
    name = "realtime_weather"
    description = "Fake weather tool"
    args = {"city": "string"}

    def __init__(self):
        self.invocations = []

    def invoke(self, args):
        self.invocations.append(args)
        return '{"ok": true, "data": {"city": "shanghai", "temp": 25}}'


class ToolCallFlowTests(unittest.TestCase):
    def test_tool_call_executes_and_returns_response(self):
        llm = FakeLLM(
            [
                'Action: realtime_weather\nAction Input: {"city": "shanghai"}',
                "Shanghai is 25°C and sunny right now.",
            ]
        )
        tool = FakeTool()
        runner = LangGraphAgentRunner(
            llm=llm,
            tools=[tool],
            system_prompt="Test prompt",
        )
        messages = [{"role": "user", "content": "What's the weather in shanghai?"}]
        output = runner.invoke(messages)

        self.assertEqual(len(tool.invocations), 1)
        self.assertEqual(tool.invocations[0], {"city": "shanghai"})

        self.assertEqual(output[-1]["role"], "assistant")
        self.assertIn("Shanghai is 25°C", output[-1]["content"])

    def test_tool_call_parses_trailing_punctuation(self):
        llm = FakeLLM(
            [
                'Action: realtime_weather\nAction Input: {"city": "shanghai"}.',
                "Done.",
            ]
        )
        tool = FakeTool()
        runner = LangGraphAgentRunner(
            llm=llm,
            tools=[tool],
            system_prompt="Test prompt",
        )
        messages = [{"role": "user", "content": "Weather in shanghai"}]
        _ = runner.invoke(messages)

        self.assertEqual(tool.invocations, [{"city": "shanghai"}])


if __name__ == "__main__":
    unittest.main()
