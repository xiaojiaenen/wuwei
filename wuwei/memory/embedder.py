from __future__ import annotations

import math
from typing import Protocol


class Embedder(Protocol):
    """Embedding 协议，任何 Embedding 服务都实现这个接口。"""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量文本转向量。"""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """单条查询转向量。"""
        results = await self.embed_texts([text])
        return results[0]


class OpenAIEmbedder:
    """OpenAI Embedding 适配器。"""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
    ):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]


class SimpleEmbedder:
    """零依赖的简易 Embedder，基于字符 n-gram 哈希。仅供测试和演示。"""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _text_to_vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        text = text.lower().strip()
        for i in range(len(text) - 1):
            bigram = text[i : i + 2]
            h = hash(bigram) % self.dim
            vec[h] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._text_to_vec(text)
