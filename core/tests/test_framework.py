from pathlib import Path
from typing import List
from langchain_ollama import ChatOllama
import pytest

from core.application.input_port import ArxivSearcher
from core.application.usecase import Searcher
from core.domain.vo import Abstract, Result
from core.framework.driven_adaptor import ArxivSearcherDrivenAdaptor
from core.framework.driving_adaptor import ArxivAgent


@pytest.fixture(scope="module")
def searcher() -> Searcher:
    searcher = ArxivSearcher(
        arxiv_searcher_output_port=ArxivSearcherDrivenAdaptor(),
        max_documents=5,
        model=ChatOllama(model="llama3.1", temperature=0.52, num_predict=500),
        prompt_templates={
            "response": Path("src/resources/searcher/response_template_v1.yaml"),
        },
    )
    return searcher


def test_arxiv_agent(searcher: Searcher):
    # given
    agent: ArxivAgent = ArxivAgent(searcher)
    query: str = "What is prompt engineering?"
    # when
    result: Result[List[Abstract]] = agent.search(query)
    # then
    assert result.answer.strip() != ""
    assert len(result.data) > 0 and len(result.data) <= 5
