from functools import partial
from typing import List
from core.application.input_port import ArxivSearcher
from core.application.output_port import response, searcher
from core.domain.vo import Abstract, Result


def test_arxiv_searcher():
    # given
    arxiv_searcher = ArxivSearcher(
        partial(searcher, n=3),
        response,
    )
    query: str = "What is Prompt Engineering?"
    # when
    result: Result[List[Abstract]] = arxiv_searcher.search(query)
    # then
    assert result != ""
    assert isinstance(result.data, list) and all(
        isinstance(abstract, Abstract) for abstract in result.data
    )
