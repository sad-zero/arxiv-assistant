from enum import Enum, unique


@unique
class ModelType(Enum):
    OLLAMA_LLAMA3_1 = "[ollama] llama3.1"
    OPENAI_GPT4o_mini = "[openai] gpt-4o-mini"
    OPENAI_GPT3_5_turbo = "[openai] gpt-3.5-turbo"
