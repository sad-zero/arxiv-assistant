from abc import ABC, abstractmethod
from typing import List

from core.domain.vo import Abstract, Result


class Searcher(ABC):
    @abstractmethod
    def search(self, query: str) -> Result[List[Abstract]]:
        pass
