from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMBackend(Protocol):
    def complete(self, prompt: str, system: str, temperature: float) -> str:
        ...


@dataclass
class ExistingRepoLLMBackend:
    """Adapter for the existing data-agent dgm_agent.llm module."""

    model: str

    def __post_init__(self) -> None:
        from dgm_agent.llm import create_client

        self.client, self.client_model = create_client(self.model)

    def complete(self, prompt: str, system: str, temperature: float) -> str:
        from dgm_agent.llm import get_response_from_llm

        content, _history = get_response_from_llm(
            prompt,
            self.client,
            self.client_model,
            system,
            print_debug=False,
            msg_history=None,
            temperature=temperature,
        )
        return content
