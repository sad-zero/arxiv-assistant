from dataclasses import dataclass
from datetime import datetime
from typing import Generic, List, TypeVar, TypedDict


from core.domain.util import validate_url


@dataclass(frozen=True, slots=True)
class Abstract:
    title: str
    abstract: str
    link: str
    written_at: datetime

    def __post_init__(self):
        assert isinstance(self.title, str)
        assert isinstance(self.abstract, str)
        assert self.link is not None and validate_url(self.link)
        assert isinstance(self.written_at, datetime)


@dataclass(frozen=True, slots=True)
class Page:
    text: str
    index: int

    def __post_init__(self):
        assert isinstance(self.text, str)
        assert isinstance(self.index, int)


@dataclass(frozen=True, slots=True)
class Document:
    abstract: Abstract
    pages: List[Page]

    def __post_init__(self):
        assert isinstance(self.abstract, Abstract)
        assert isinstance(self.pages, list) and all(
            isinstance(page, Page) for page in self.pages
        )

    @property
    def num_page(self) -> int:
        return len(self.pages)


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Result(Generic[T]):
    answer: str
    data: T


class SearchState(TypedDict):
    query: str
    answer: str
    abstracts: List[Abstract]
