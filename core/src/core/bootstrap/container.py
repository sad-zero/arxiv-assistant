"""Build Assistant"""

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Literal, Union

from dependency_injector.containers import DeclarativeContainer
from dependency_injector.providers import Configuration, Dependency, Singleton
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from core.application.input_port import ArxivSearcher
from core.application.output_port import ArxivSearcherOutputPort
from core.bootstrap.vo import ArxivSearcherConfig, AzureConfig
from core.framework.driven_adaptor import ArxivSearcherDrivenAdaptor
from core.framework.driving_adaptor import ArxivAgent
from core.framework.vo import ModelType


class ModelContainer(DeclarativeContainer):
    config = Configuration()

    ollama_llama3_1: ChatOllama = Singleton(
        ChatOllama,
        model="llama3.1",
        temperature=0.52,
        num_predicts=500,
    )
    openai_gpt4o_mini: ChatOpenAI = Singleton(
        ChatOpenAI,
        model="gpt-4o-mini",
        temperature=0.52,
        max_tokens=500,
        api_key=config.openai.api_key,
    )
    openai_gpt3_5_turbo: ChatOpenAI = Singleton(
        ChatOpenAI,
        model="gpt-3.5-turbo",
        temperature=0.52,
        max_tokens=500,
        api_key=config.openai.api_key,
    )
    azure_model = Singleton(
        AzureChatOpenAI,
        azure_endpoint=config.azure.endpoint,
        azure_deployment=config.azure.deployment,
        api_version=config.azure.api_version,
        api_key=config.azure.api_key,
        temperature=0.52,
        max_tokens=500,
    )


class ApplicationOutputPortContainer(DeclarativeContainer):
    arxiv_searcher_output_port: ArxivSearcherOutputPort = Singleton(
        ArxivSearcherDrivenAdaptor
    )


class ApplicationUsecaseContainer(DeclarativeContainer):
    config = Configuration()
    arxiv_searcher_output_port: ArxivSearcherOutputPort = Dependency(
        instance_of=ArxivSearcherOutputPort,
    )

    arxiv_searcher: ArxivSearcher = Singleton(
        ArxivSearcher,
        arxiv_searcher_output_port=arxiv_searcher_output_port,
        max_documents=config.arxiv_searcher.max_documents,
        model=config.arxiv_searcher.model,
        prompt_templates=config.arxiv_searcher.prompt_templates,
    )


class FrameworkDrivingContainer(DeclarativeContainer):
    arxiv_searcher: ArxivSearcher = Dependency(instance_of=ArxivSearcher)

    arxiv_agent: ArxivAgent = Singleton(
        ArxivAgent,
        searcher=arxiv_searcher,
    )


def get_prompt_templates(type_: Literal["arxiv_searcher"]) -> Dict[str, Path]:
    assert type_ in {
        "arxiv_searcher",
    }

    template_base: Path = Path(__file__).parent.parent.parent / "resources"
    templates_by_types: Dict[Literal["arxiv_searcher"], Dict[str, Path]] = {
        "arxiv_searcher": {
            "response": template_base / "searcher/response_template_v1.yaml",
        }
    }
    return templates_by_types[type_]


def initialize(
    configs: Dict[
        Literal["arxiv_searcher", "azure"], Union[ArxivSearcherConfig, AzureConfig]
    ],
    openai_key: str,
) -> ArxivAgent:
    assert isinstance(configs, dict)
    assert "arxiv_searcher" in configs and isinstance(
        configs["arxiv_searcher"], ArxivSearcherConfig
    )

    if "azure" in configs:
        assert isinstance(configs["azure"], AzureConfig)
    assert isinstance(openai_key, str)

    model_container = ModelContainer()
    model_container.config.from_dict(
        {
            "openai": {
                "api_key": openai_key,
            },
            "azure": asdict(configs["azure"]),
        }
    )

    application_output_port_container = ApplicationOutputPortContainer()
    application_output_port_container.check_dependencies()
    application_usecase_container = ApplicationUsecaseContainer(
        arxiv_searcher_output_port=application_output_port_container.arxiv_searcher_output_port(),
    )
    application_usecase_container.check_dependencies()

    arxiv_searcher_config = configs["arxiv_searcher"].to_dict()
    arxiv_searcher_config["prompt_templates"] = get_prompt_templates("arxiv_searcher")
    if arxiv_searcher_config["model_type"] == ModelType.OLLAMA_LLAMA3_1:
        arxiv_searcher_config["model"] = model_container.ollama_llama3_1()
    elif arxiv_searcher_config["model_type"] == ModelType.OPENAI_GPT4o_mini:
        arxiv_searcher_config["model"] = model_container.openai_gpt4o_mini()
    elif arxiv_searcher_config["model_type"] == ModelType.OPENAI_GPT3_5_turbo:
        arxiv_searcher_config["model"] = model_container.openai_gpt3_5_turbo()
    elif arxiv_searcher_config["model_type"] == ModelType.AZURE_AI_STUDIO:
        arxiv_searcher_config["model"] = model_container.azure_model()
    else:
        raise RuntimeError(
            f"Not supported model type: {arxiv_searcher_config['model_type']}"
        )

    application_usecase_container.config.from_dict(
        {
            "arxiv_searcher": arxiv_searcher_config,
        }
    )

    framework_driving_container = FrameworkDrivingContainer(
        arxiv_searcher=application_usecase_container.arxiv_searcher(),
    )
    framework_driving_container.check_dependencies()
    return framework_driving_container.arxiv_agent()
