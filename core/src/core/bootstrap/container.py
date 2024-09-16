"""Build Assistant"""

from pathlib import Path
from typing import Any, Dict, Literal, Union

from dependency_injector.containers import DeclarativeContainer
from dependency_injector.providers import Configuration, Dependency, Singleton
from core.application.input_port import ArxivSearcher
from core.application.output_port import ArxivSearcherOutputPort
from core.bootstrap.vo import ArxivSearcherConfig
from core.framework.driven_adaptor import ArxivSearcherDrivenAdaptor
from core.framework.driving_adaptor import ArxivAgent


class ApplicationOutputPortContainer(DeclarativeContainer):
    arxiv_searcher_output_port: ArxivSearcherOutputPort = Singleton(
        ArxivSearcherDrivenAdaptor
    )


class ApplicationUsecaseContainer(DeclarativeContainer):
    config = Configuration()
    arxiv_searcher_output_port: ArxivSearcherOutputPort = Dependency(
        ArxivSearcherOutputPort
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
    configs: Dict[Literal["arxiv_searcher"], Union[ArxivSearcherConfig]],
) -> ArxivAgent:
    assert isinstance(configs, dict)
    assert "arxiv_searcher" in configs and isinstance(
        configs["arxiv_searcher"], ArxivSearcherConfig
    )

    application_output_port_container = ApplicationOutputPortContainer()
    application_output_port_container.check_dependencies()

    application_usecase_container = ApplicationUsecaseContainer(
        arxiv_searcher_output_port=application_output_port_container.arxiv_searcher_output_port(),
    )
    application_usecase_container.check_dependencies()

    config_dictionaries: Dict[Literal["arxiv_searcher"], Dict[str, Any]] = {
        key: val.to_dict() for key, val in configs.items()
    }
    config_dictionaries["arxiv_searcher"]["prompt_templates"] = get_prompt_templates(
        "arxiv_searcher"
    )
    application_usecase_container.config.from_dict(config_dictionaries)

    framework_driving_container = FrameworkDrivingContainer(
        arxiv_searcher=application_usecase_container.arxiv_searcher(),
    )
    framework_driving_container.check_dependencies()
    return framework_driving_container.arxiv_agent()
