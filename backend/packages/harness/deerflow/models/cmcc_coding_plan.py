"""China Mobile eCloud Coding Plan OpenAI-compatible adapter.

Coding Plan documents an OpenAI-compatible /chat/completions endpoint, but
some coding-tool gateways are stricter about optional request fields than the
OpenAI API. This adapter keeps the normal ChatOpenAI behavior while trimming
DeerFlow/LangChain metadata that can cause schema rejection.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_openai import ChatOpenAI


_DROP_ALWAYS = {
    # Optional OpenAI extensions that are not required for agent operation and
    # are often rejected by smaller OpenAI-compatible gateways.
    "parallel_tool_calls",
    "stream_options",
    "reasoning_effort",
    "service_tier",
    "store",
    "metadata",
}


def _normalize_coding_plan_payload(payload: dict[str, Any]) -> None:
    """Make LangChain's request body conservative for Coding Plan."""
    for key in list(payload.keys()):
        if payload[key] is None:
            payload.pop(key, None)

    for key in _DROP_ALWAYS:
        payload.pop(key, None)

    if payload.get("tool_choice") == "any":
        payload["tool_choice"] = "required"

    tools = payload.get("tools")
    if tools == []:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return

    for message in messages:
        if not isinstance(message, dict):
            continue

        # Older OpenAI-compatible gateways commonly only accept system/user/
        # assistant/tool/function roles.
        if message.get("role") == "developer":
            message["role"] = "system"

        # DeerFlow can attach internal provenance names to messages. Coding Plan
        # does not need them, and provider-side validation may reject them.
        if message.get("role") != "function":
            message.pop("name", None)

        for key in list(message.keys()):
            if message[key] is None:
                message.pop(key, None)


class ChinaMobileCodingPlanChatModel(ChatOpenAI):
    """ChatOpenAI wrapper for China Mobile eCloud Coding Plan."""

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "cmcc-coding-plan-openai-compatible"

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        _normalize_coding_plan_payload(payload)
        return payload
