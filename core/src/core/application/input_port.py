from pathlib import Path
from typing import Callable, Dict, List

from langchain_core.language_models import BaseChatModel
from core.application.output_port import ArxivSearcherOutputPort
from core.application.usecase import Searcher
from core.domain.entity import SearchWorkflow
from core.domain.vo import Abstract, Result, SearchState


class ArxivSearcher(Searcher):
    def __init__(
        self,
        arxiv_searcher_output_port: ArxivSearcherOutputPort,
        max_documents: int,
        model: BaseChatModel,
        prompt_templates: Dict[str, Path],
    ):
        assert isinstance(arxiv_searcher_output_port, ArxivSearcherOutputPort)
        assert isinstance(max_documents, int) and max_documents > 0
        assert isinstance(model, BaseChatModel)
        assert isinstance(prompt_templates, dict) and all(
            isinstance(key, str) and isinstance(value, Path) and value.exists()
            for key, value in prompt_templates.items()
        )
        assert "response" in prompt_templates

        searcher: Callable[[SearchState], SearchState] = (
            arxiv_searcher_output_port.searcher(max_documents)
        )
        response: Callable[[SearchState], SearchState] = (
            arxiv_searcher_output_port.response(model, prompt_templates["response"])
        )
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
