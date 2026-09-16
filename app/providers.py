"""
Thin provider abstraction so app/agent.py doesn't care whether it's talking
to Anthropic or Gemini. Tool schema is defined once, in Anthropic's shape
(TOOLS in app/agent.py), and converted per-provider here.

Both providers expose the same interface:
    provider.step(messages) -> StepResult
    provider.append_assistant_turn(messages, step_result) -> messages
    provider.append_tool_results(messages, step_result, results) -> messages

`messages` is kept in the provider's own native format inside each Session --
that's fine because the provider is fixed for the lifetime of a process via
LLM_PROVIDER, sessions are never migrated between providers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRequest:
    call_id: str
    name: str
    input: dict[str, Any]


@dataclass
class StepResult:
    text: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    raw: Any = None  # provider-native response object, needed to append history correctly


class LLMProvider:
    def step(self, system_prompt: str, tools: list[dict], messages: list[Any]) -> StepResult:
        raise NotImplementedError

    def append_assistant_turn(self, messages: list[Any], result: StepResult) -> list[Any]:
        raise NotImplementedError

    def append_tool_results(
        self, messages: list[Any], result: StepResult, results: list[dict[str, Any]]
    ) -> list[Any]:
        raise NotImplementedError

    def append_user_message(self, messages: list[Any], text: str) -> list[Any]:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------
class AnthropicProvider(LLMProvider):
    def __init__(self, model: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    def step(self, system_prompt, tools, messages) -> StepResult:
        anth_tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
            for t in tools
        ]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system_prompt,
            tools=anth_tools,
            messages=messages,
        )
        text_blocks = [b.text for b in response.content if b.type == "text"]
        tool_calls = [
            ToolCallRequest(call_id=b.id, name=b.name, input=b.input)
            for b in response.content
            if b.type == "tool_use"
        ]
        return StepResult(text="\n".join(text_blocks), tool_calls=tool_calls, raw=response)

    def append_assistant_turn(self, messages, result: StepResult):
        return messages + [{"role": "assistant", "content": result.raw.content}]

    def append_tool_results(self, messages, result: StepResult, results):
        content = [
            {"type": "tool_result", "tool_use_id": tc.call_id, "content": str(res)}
            for tc, res in zip(result.tool_calls, results)
        ]
        return messages + [{"role": "user", "content": content}]

    def append_user_message(self, messages, text):
        return messages + [{"role": "user", "content": text}]


# --------------------------------------------------------------------------
# Gemini (free tier via Google AI Studio key)
# --------------------------------------------------------------------------
class GeminiProvider(LLMProvider):
    def __init__(self, model: str | None = None):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    @staticmethod
    def _json_schema_to_gemini(schema: dict) -> dict:
        # Gemini's function-declaration schema is close enough to plain JSON
        # Schema for this project's simple string-only tool; pass through.
        return schema

    def _build_tools(self, tools: list[dict]):
        from google.genai import types

        decls = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=self._json_schema_to_gemini(t["input_schema"]),
            )
            for t in tools
        ]
        return [types.Tool(function_declarations=decls)]

    def step(self, system_prompt, tools, messages) -> StepResult:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=self._build_tools(tools) if tools else None,
        )
        response = self.client.models.generate_content(
            model=self.model, contents=messages, config=config
        )
        candidate = response.candidates[0]
        parts = candidate.content.parts or []
        text = "".join(p.text for p in parts if getattr(p, "text", None))
        tool_calls = [
            ToolCallRequest(call_id=p.function_call.name, name=p.function_call.name, input=dict(p.function_call.args))
            for p in parts
            if getattr(p, "function_call", None)
        ]
        return StepResult(text=text, tool_calls=tool_calls, raw=response)

    def append_assistant_turn(self, messages, result: StepResult):
        return messages + [result.raw.candidates[0].content]

    def append_tool_results(self, messages, result: StepResult, results):
        from google.genai import types

        parts = [
            types.Part.from_function_response(name=tc.name, response={"result": res})
            for tc, res in zip(result.tool_calls, results)
        ]
        return messages + [types.Content(role="user", parts=parts)]

    def append_user_message(self, messages, text):
        from google.genai import types

        return messages + [types.Content(role="user", parts=[types.Part.from_text(text=text)])]


def build_provider() -> LLMProvider:
    provider_name = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider_name == "gemini":
        return GeminiProvider()
    return AnthropicProvider()
