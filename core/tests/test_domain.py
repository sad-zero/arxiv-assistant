from datetime import datetime
from typing import List
import pytest

from core.domain.entity import SearchWorkflow
from core.domain.util import validate_url
from core.domain.vo import Abstract, Result, SearchState


@pytest.mark.parametrize(
    "url",
    [
        "https://www.example.com",
        "http://localhost:8000/test",
        "http://192.168.0.1:8080/test",
    ],
)
def test_validate_url(url: str):
    # Example usage
    assert validate_url(url)


def test_search_workflow():
    # given
    abstracts = [
        Abstract(
            title="sample",
            abstract="sample",
            link="https://test.com",
            written_at=datetime.now(),
        ),
    ]

    def searcher(state: SearchState) -> SearchState:
        return {
            "abstracts": abstracts,
        }

    def response(state: SearchState) -> SearchState:
        answer = f"Query: {state['query']}"
        return {
            "answer": answer,
        }

    workflow = SearchWorkflow.builder().searcher(searcher).response(response).build()
    query: str = "Sample"

    # when
    result: Result[List[Abstract]] = workflow.execute(query)
    # then
    assert f"Query: {query}" == result.answer
    assert abstracts == result.data
