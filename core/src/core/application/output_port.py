from pathlib import Path
from typing import List
from langchain.prompts import ChatPromptTemplate, PromptTemplate, load_prompt
from langchain_community.retrievers import ArxivRetriever
from langchain.schema import Document as LangchainDocument, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from core.application.util import OllamaCallbacks
from core.domain.vo import Abstract, SearchState


def searcher(state: SearchState, n: int = 5) -> SearchState:
    retriever = ArxivRetriever(
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


def response(state: SearchState, template_path: Path) -> SearchState:
    assert isinstance(template_path, Path) and template_path.exists()

    model = ChatOllama(model="llama3.1", temperature=0.52, num_predict=500)
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
        {"abstracts": lambda _: state["abstracts"], "query": RunnablePassthrough()}
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
