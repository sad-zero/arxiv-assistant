"""Build Assistant"""

from functools import partial
from pathlib import Path
from core.application.input_port import ArxivSearcher
from core.application.output_port import response, searcher


def get_arxiv_searcher():
    resource_path = Path(__file__).parent.parent / "resources"

    searcher_response_template_path = (
        resource_path / "searcher/response_template_v1.yaml"
    )
    arxiv_searcher = ArxivSearcher(
        searcher=partial(searcher, n=5),
        response=partial(response, template_path=searcher_response_template_path),
    )
    return arxiv_searcher
