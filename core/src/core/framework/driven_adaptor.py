from pathlib import Path
from typing import Callable, List
from langchain.prompts import ChatPromptTemplate, PromptTemplate, load_prompt
from langchain_community.retrievers import ArxivRetriever
from langchain.schema import Document as LangchainDocument, StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnablePassthrough
from core.application.output_port import ArxivSearcherOutputPort
from core.framework.util import OllamaCallbacks
from core.domain.vo import Abstract, SearchState


class ArxivSearcherDrivenAdaptor(ArxivSearcherOutputPort):
    def searcher(self, n: int) -> Callable[[SearchState], SearchState]:
        assert isinstance(n, int) and n > 0

        def _searcher(state: SearchState) -> SearchState:
            retriever = ArxivRetriever(
                top_k_results=n,
                load_max_docs=n,
                get_full_documents=False,
            )
            documents: List[LangchainDocument] = retriever.invoke(state["query"])
            abstracts: List[Abstract] = [
                Abstract(
                    title=document.metadata["Title"],
                    abstract=document.page_content,
                    link=document.metadata["Entry ID"],
                    written_at=document.metadata["Published"],
                )
                for document in documents
            ]
            return {
                "abstracts": abstracts,
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
            response: str = chain.invoke(
                state["query"], config={"callbacks": [OllamaCallbacks()]}
            )
            return {
                "answer": response,
            }

        return _response
