import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_CHAT_MODEL = os.getenv("NVIDIA_CHAT_MODEL", "meta/llama-3.1-70b-instruct")
NVIDIA_EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")


def _get_api_key() -> str:
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set NVIDIA_API_KEY in .env to use the NVIDIA API.")
    return api_key


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=_get_api_key())


def chat_completion(
    prompt: str,
    *,
    system_instruction: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
) -> str:
    response = get_client().chat.completions.create(
        model=model or NVIDIA_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    content = response.choices[0].message.content or ""
    return content.strip()


def stream_chat_completion(
    prompt: str,
    *,
    system_instruction: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
):
    stream = get_client().chat.completions.create(
        model=model or NVIDIA_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_output_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def embed_texts(texts: list[str], *, model: str | None = None, input_type: str = "passage") -> list[list[float]]:
    response = get_client().embeddings.create(
        model=model or NVIDIA_EMBED_MODEL,
        input=texts,
        extra_body={"input_type": input_type},
    )
    return [item.embedding for item in response.data]
