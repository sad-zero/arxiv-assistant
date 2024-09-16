from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel


@dataclass(frozen=True, slots=True)
class ArxivSearcherConfig:
    max_documents: int
    model: BaseChatModel

    def __post_init__(self):
        assert isinstance(self.max_documents, int) and self.max_documents > 0
        assert isinstance(self.model, BaseChatModel)

    def to_dict(self) -> dict:
        return {
            "max_documents": self.max_documents,
            "model": self.model,
        }
