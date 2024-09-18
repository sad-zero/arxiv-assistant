from enum import Enum, auto, unique


@unique
class ModelType(Enum):
    OLLAMA_LLAMA3_1 = auto()
    OPENAI_GPT4o = auto()
