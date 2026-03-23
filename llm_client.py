"""
General-purpose LLM client using AWS Bedrock.

Usage:
    from llm_client import call_llm, LLMClient

    # Simple one-off call
    response = call_llm("What is value reframing?")

    # With system prompt
    response = call_llm(
        user_message="Reframe this statement: ...",
        system_prompt="You are an expert in value framing.",
    )

    # Reusable client with custom settings
    client = LLMClient(model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    response = client.chat("Hello!")
"""

import os
import boto3
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL_ID = os.getenv(
    "AWS_BEDROCK_MODEL_ID",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
)
DEFAULT_REGION = os.getenv("AWS_BEDROCK_REGION", "us-east-1")


class LLMClient:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        region: str = DEFAULT_REGION,
    ):
        api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        if not api_key:
            raise EnvironmentError(
                "AWS_BEARER_TOKEN_BEDROCK is not set. "
                "Copy .env.example to .env and fill in your API key."
            )
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key

        self.model_id = model_id
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
        )

    @staticmethod
    def _extract_text(content: list[dict]) -> str:
        """Extract text from Bedrock converse response content blocks.

        Handles standard text blocks and reasoning model formats (e.g., DeepSeek R1).
        """
        for block in content:
            if "text" in block:
                return block["text"]
        # Fallback: reasoning-only response
        for block in content:
            if "reasoningContent" in block:
                return block["reasoningContent"]["reasoningText"]["text"]
        return content[0].get("text", str(content[0]))

    def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send a single user message and return the assistant's text reply."""
        messages = [
            {"role": "user", "content": [{"text": user_message}]}
        ]

        kwargs: dict = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }

        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        response = self.client.converse(**kwargs)
        return self._extract_text(response["output"]["message"]["content"])

    def chat_with_history(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Send a full conversation history and return the assistant's reply.

        messages format:
            [
                {"role": "user", "content": [{"text": "..."}]},
                {"role": "assistant", "content": [{"text": "..."}]},
                ...
            ]
        """
        kwargs: dict = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }

        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        response = self.client.converse(**kwargs)
        return self._extract_text(response["output"]["message"]["content"])


# Module-level singleton — shared across the project
_default_client: LLMClient | None = None


def _get_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def call_llm(
    user_message: str,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    model_id: str = DEFAULT_MODEL_ID,
    region: str = DEFAULT_REGION,
) -> str:
    """Convenience function for a single-turn LLM call.

    Uses a shared client when model/region match the defaults,
    otherwise creates a temporary client with the given settings.
    """
    if model_id == DEFAULT_MODEL_ID and region == DEFAULT_REGION:
        client = _get_client()
    else:
        client = LLMClient(model_id=model_id, region=region)

    return client.chat(
        user_message=user_message,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


if __name__ == "__main__":
    print(call_llm("Say hello in one sentence."))
