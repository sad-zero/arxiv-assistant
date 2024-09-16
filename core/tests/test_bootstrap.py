from langchain_ollama import ChatOllama
from core.bootstrap.container import initialize
from core.bootstrap.vo import ArxivSearcherConfig
from core.framework.driving_adaptor import ArxivAgent


def test_bootstrap():
    # given
    arxiv_searcher_config = ArxivSearcherConfig(
        max_documents=5,
        model=ChatOllama(model="llama3.1", temperature=0.52, num_predict=500),
    )
    # when
    arxiv_agent: ArxivAgent = initialize(
        configs={
            "arxiv_searcher": arxiv_searcher_config,
        }
    )
    # then
    assert isinstance(arxiv_agent, ArxivAgent)
