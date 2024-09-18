from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from langchain_core.language_models import BaseChatModel

from core.domain.vo import SearchState


class ArxivSearcherOutputPort(ABC):
    @abstractmethod
    def searcher(self, n: int) -> Callable[[SearchState], SearchState]:
        """Return Searcher
        Args:
            n(int): The number of retrieved documents
        Returns:
            Callable[[SearchState], SearchState]: searcher function
        """
        pass

    @abstractmethod
    def response(
        self, model: BaseChatModel, template_path: Path
    ) -> Callable[[SearchState], SearchState]:
        """Return Response generator
        Args:
            model(BaseChatModel): Chat LLM
            template_path(str): Prompt Template Path
        Returns:
            Callable[[SearchState], SearchState]: response function
        """
        pass
