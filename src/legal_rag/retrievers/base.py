from typing import Protocol

from legal_rag.types import RetrievalResult


class BaseRetriever(Protocol):
    def retrieve(self, question: str, **kwargs) -> RetrievalResult:
        ...

