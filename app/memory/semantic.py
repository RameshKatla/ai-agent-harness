import math
from typing import Protocol

import httpx
from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import MemoryError
from app.db.models import EMBEDDING_DIMENSIONS


class Embedder(Protocol):
    @property
    def model(self) -> str: ...

    async def embed(self, text: str, *, query: bool) -> list[float]: ...


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]


class OllamaEmbedder:
    def __init__(self, settings: Settings, http: httpx.AsyncClient) -> None:
        self.model = settings.ollama_embedding_model
        self._url = str(settings.ollama_base_url).rstrip("/") + "/api/embed"
        self._http = http
        self._timeout = httpx.Timeout(settings.agent_timeout_seconds, connect=5)

    async def embed(self, text: str, *, query: bool) -> list[float]:
        if self.model.partition(":")[0] == "nomic-embed-text":
            prefix = "search_query" if query else "search_document"
            text = f"{prefix}: {text}"
        try:
            response = await self._http.post(
                self._url,
                json={"model": self.model, "input": text, "dimensions": EMBEDDING_DIMENSIONS},
                timeout=self._timeout,
            )
            response.raise_for_status()
            vectors = EmbeddingResponse.model_validate_json(response.content).embeddings
        except (httpx.HTTPError, ValueError):
            raise MemoryError("Could not generate a memory embedding.") from None

        if len(vectors) != 1:
            raise MemoryError("The embedding service returned an unexpected vector count.")
        vector = vectors[0]
        if (
            len(vector) != EMBEDDING_DIMENSIONS
            or not all(math.isfinite(value) and abs(value) <= 1 for value in vector)
            or math.hypot(*vector) < 1e-12
        ):
            raise MemoryError("The embedding service returned an invalid vector.")
        return vector
