from abc import ABC, abstractmethod
import logging
from typing import Callable, Generic, List, TypeVar

from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph

from core.domain.vo import Abstract, Result, SearchState

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

T = TypeVar("T")


class Workflow(ABC, Generic[T]):
    """Define Workflow"""

    @abstractmethod
    def execute(self, query: str) -> Result[T]:
        """Execute workflow and Return result

        Args:
            query(str): Human's query
        Returns:
            Result: Return workflow result
        """
        pass

    @classmethod
    @abstractmethod
    def builder(cls) -> "Workflow[T]":
        pass


class SearchWorkflow(Workflow[List[Abstract]]):
    def __init__(self, engine: CompiledGraph):
        assert isinstance(engine, CompiledGraph)
        self.__engine: CompiledGraph = engine

    def execute(self, query: str) -> Result[List[Abstract]]:
        init_state: SearchState = {
            "query": query,
        }
        logger.info(f"Initial State: {init_state}")
        final_state: SearchState = self.__engine.invoke(init_state)
        logger.info(f"Final State: {final_state}")
        result = Result(answer=final_state["answer"], data=final_state["abstracts"])
        return result

    @classmethod
    def builder(cls) -> "SearchWorkflow.Builder":
        return SearchWorkflow.Builder()

    class Builder:
        def __init__(self):
            self.__graph = StateGraph(SearchState)

        def searcher(
            self, searcher: Callable[[SearchState], SearchState]
        ) -> "SearchWorkflow.Builder":
            self.__graph.add_node("searcher", searcher)
            return self

        def response(
            self, response: Callable[[SearchState], SearchState]
        ) -> "SearchWorkflow.Builder":
            self.__graph.add_node("response", response)
            return self

        def build(self) -> "SearchWorkflow":
            try:
                self.__graph.add_edge(START, "searcher")
                self.__graph.add_edge("searcher", "response")
                self.__graph.add_edge("response", END)
                engine: CompiledGraph = self.__graph.compile()
            except ValueError as e:
                raise RuntimeError("Cannot build SearchWorkflow.") from e
            else:
                return SearchWorkflow(engine)
