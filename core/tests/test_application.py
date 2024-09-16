from datetime import datetime
from pathlib import Path
from typing import Callable, List

from langchain.prompts import ChatPromptTemplate, PromptTemplate, load_prompt
from langchain.schema import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
import pytest
from core.application.input_port import ArxivSearcher
from core.application.output_port import ArxivSearcherOutputPort
from core.domain.vo import Abstract, Result, SearchState


class StubArxivSearcherOutputPort(ArxivSearcherOutputPort):
    def searcher(self, n: int) -> Callable[[SearchState], SearchState]:
        abstract = Abstract(
            title="test",
            abstract="test",
            link="https://test.com",
            written_at=datetime.now(),
        )

        def _searcher(state: SearchState) -> SearchState:
            return {
                "abstracts": [abstract for _ in range(n)],
            }

        return _searcher

    def response(
        self, model: BaseChatModel, template_path: Path
    ) -> Callable[[SearchState], SearchState]:
        assert isinstance(model, BaseChatModel)
        assert isinstance(template_path, Path) and template_path.exists()

        def _response(state: SearchState) -> SearchState:
            system_prompt: PromptTemplate = load_prompt(template_path)
            template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt.template),
                    (
                        "human",
                        """
        query: {query}
        abstracts: {abstracts}

        agent.answer_with_abstracts(query, abstracts)
        """.strip(),
                    ),
                ]
            )
            chain = (
                {
                    "abstracts": lambda _: state["abstracts"],
                    "query": RunnablePassthrough(),
                }
                | template
                | model
                | StrOutputParser()
            )
            response: str = chain.invoke(state["query"])
            return {
                "answer": response,
            }

        return _response


@pytest.fixture(scope="module")
def arxiv_searcher_output_port() -> ArxivSearcherOutputPort:
    return StubArxivSearcherOutputPort()


def test_arxiv_searcher(arxiv_searcher_output_port: ArxivSearcherOutputPort):
    # given
    max_documents = 5
    model = ChatOllama(model="llama3.1", temperature=0.52, num_predict=500)
    arxiv_searcher = ArxivSearcher(
        arxiv_searcher_output_port=arxiv_searcher_output_port,
        max_documents=max_documents,
        model=model,
        prompt_templates={
            "response": Path("src/resources/searcher/response_template_v1.yaml"),
        },
    )
    query: str = "What is Prompt Engineering?"
    # when
    result: Result[List[Abstract]] = arxiv_searcher.search(query)
    # then
    assert result.answer.strip() != ""
    assert isinstance(result.data, list) and all(
        isinstance(abstract, Abstract) for abstract in result.data
    )
    assert len(result.data) <= max_documents
