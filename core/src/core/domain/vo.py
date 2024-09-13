from dataclasses import dataclass
from datetime import datetime

from core.domain.util import validate_url


@dataclass(frozen=True, slots=True)
class Document:
    title: str
    abstract: str
    link: str
    written_at: datetime

    def __post_init__(self):
        assert self.title is not None and isinstance(self.title, str)
        assert self.abstract is not None and isinstance(self.abstract, str)
        assert self.link is not None and validate_url(self.link)
        assert self.written_at is not None and isinstance(self.written_at, datetime)
