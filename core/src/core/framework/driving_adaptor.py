import logging
from typing import List

from core.application.usecase import Searcher
from core.domain.vo import Abstract, Result

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ArxivAgent:
    def __init__(self, searcher: Searcher):
        assert isinstance(searcher, Searcher)

        self.__searcher: Searcher = searcher

    def search(self, query: str) -> Result[List[Abstract]]:
        """Search Arxiv Abstracts and Return result
        Args:
            query(str): User's query
        Returns:
            Result[List[Abstract]]: Agent's answer with related papers
        """
        try:
            result: Result[List[Abstract]] = self.__searcher.search(query)
        except RuntimeError as e:
            logger.error(e)
            return Result(
                answer=f'Sorry, I can\'t answer about "{query}". Please check log',
                data=[],
            )
        else:
            return result
