from datetime import datetime
from typing import List
from core.application.input_port import ArxivSearcher
from core.domain.vo import Abstract, Result, SearchState


def searcher(state: SearchState) -> SearchState:
    return {
        "abstracts": [
            Abstract(
                title="test",
                abstract="test",
                link="https://test.com",
                written_at=datetime.now(),
            ),
        ]
    }


def response(state: SearchState) -> SearchState:
    return {"answer": "test answer"}


def test_arxiv_searcher():
    # given
    arxiv_searcher = ArxivSearcher(
        searcher,
        response,
    )
    query: str = "What is Prompt Engineering?"
    # when
    result: Result[List[Abstract]] = arxiv_searcher.search(query)
    # then
    assert result.answer == "test answer"
    assert isinstance(result.data, list) and all(
        isinstance(abstract, Abstract) for abstract in result.data
    )
