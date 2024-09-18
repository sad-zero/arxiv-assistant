from dataclasses import dataclass


from core.framework.vo import ModelType


@dataclass(frozen=True, slots=True)
class ArxivSearcherConfig:
    max_documents: int
    model_type: ModelType

    def __post_init__(self):
        assert isinstance(self.max_documents, int) and self.max_documents > 0
        assert isinstance(self.model_type, ModelType)

    def to_dict(self) -> dict:
        return {
            "max_documents": self.max_documents,
        }
