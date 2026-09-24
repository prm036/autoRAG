"""
LLM client — unified interface for language model generation.

Supports multiple providers (OpenAI, Google Gemini, Ollama) behind a
single generate() interface. Tracks token usage and latency for every call.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from src.models import LLMResponse

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client supporting OpenAI, Gemini, and Ollama.

    Args:
        provider: "openai", "gemini", or "ollama".
        model_name: Model identifier (e.g., "gpt-4o-mini").
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        **kwargs: Provider-specific configuration.

    Usage:
        llm = LLMClient(provider="openai", model_name="gpt-4o-mini")
        response = llm.generate("What is RAG?")
        response = llm.generate(prompt, system_prompt="You are a helpful assistant.")
    """

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._kwargs = kwargs

        # Lazy-initialized client
        self._client = None

    def _get_client(self):
        """Lazy-initialize the provider client."""
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            from openai import OpenAI

            api_key = os.environ.get(
                self._kwargs.get("api_key_env", "OPENAI_API_KEY")
            )
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. "
                    "Set it with: export OPENAI_API_KEY=your-key"
                )
            self._client = OpenAI(api_key=api_key)

        elif self.provider == "gemini":
            import google.generativeai as genai

            api_key = os.environ.get(
                self._kwargs.get("api_key_env", "GEMINI_API_KEY")
            )
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable not set. "
                    "Set it with: export GEMINI_API_KEY=your-key"
                )
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.model_name)

        elif self.provider == "ollama":
            from openai import OpenAI

            base_url = self._kwargs.get("base_url", "http://localhost:11434/v1")
            self._client = OpenAI(base_url=base_url, api_key="ollama")

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        logger.info(f"Initialized {self.provider} client (model={self.model_name})")
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt / main input.
            system_prompt: Optional system-level instructions.

        Returns:
            LLMResponse with text, token counts, and latency.
        """
        start = time.time()
        client = self._get_client()

        if self.provider in ("openai", "ollama"):
            response = self._generate_openai(client, prompt, system_prompt)
        elif self.provider == "gemini":
            response = self._generate_gemini(client, prompt, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        response.latency_ms = (time.time() - start) * 1000
        response.model = self.model_name

        logger.info(
            f"LLM response: {response.input_tokens} in / "
            f"{response.output_tokens} out tokens, "
            f"{response.latency_ms:.0f}ms (model={self.model_name})"
        )

        return response

    def _generate_openai(
        self, client, prompt: str, system_prompt: str | None
    ) -> LLMResponse:
        """Generate using OpenAI-compatible API (also works for Ollama)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return LLMResponse(
            text=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    def _generate_gemini(
        self, client, prompt: str, system_prompt: str | None
    ) -> LLMResponse:
        """Generate using Google Gemini API."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        response = client.generate_content(
            full_prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        )

        # Extract token counts from usage metadata
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

        return LLMResponse(
            text=response.text if response.text else "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def info(self) -> dict:
        """Return metadata about the LLM configuration."""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
