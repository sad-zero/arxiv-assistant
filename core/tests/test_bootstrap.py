from core.bootstrap.container import initialize
from core.bootstrap.vo import ArxivSearcherConfig
from core.framework.driving_adaptor import ArxivAgent
from core.framework.vo import ModelType


def test_bootstrap():
    # given
    arxiv_searcher_config = ArxivSearcherConfig(
        max_documents=5,
        model_type=ModelType.OLLAMA_LLAMA3_1,
    )
    # when
    arxiv_agent: ArxivAgent = initialize(
        configs={
            "arxiv_searcher": arxiv_searcher_config,
        },
        openai_key="test",
    )
    # then
    assert isinstance(arxiv_agent, ArxivAgent)
