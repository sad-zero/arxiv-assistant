from typing import Callable, List
from core.application.usecase import Searcher
from core.domain.entity import SearchWorkflow
from core.domain.vo import Abstract, Result, SearchState


class ArxivSearcher(Searcher):
    def __init__(
        self,
        searcher: Callable[[SearchState], SearchState],
        response: Callable[[SearchState], SearchState],
    ):
        try:
            self.__workflow = (
                SearchWorkflow.builder().searcher(searcher).response(response).build()
            )
        except RuntimeError as e:
            raise RuntimeError("Cannot initialize ArxivSearcher") from e

    def search(self, query: str) -> Result[List[Abstract]]:
        assert isinstance(query, str)
        result: Result[List[Abstract]] = self.__workflow.execute(query)
        return result
