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
            "model_type": self.model_type,
        }


@dataclass(frozen=True, slots=True)
class AzureConfig:
    """Azure AI Studio MaaS Config"""

    endpoint: str
    deployment: str
    api_version: str
    api_key: str

    def __post_init__(self):
        assert isinstance(self.endpoint, str)
        assert isinstance(self.deployment, str)
        assert isinstance(self.api_version, str)
        assert isinstance(self.api_key, str)
