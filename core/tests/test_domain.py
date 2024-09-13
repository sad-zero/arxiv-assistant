import pytest

from core.domain.util import validate_url


@pytest.mark.parametrize(
    "url",
    [
        "https://www.example.com",
        "http://localhost:8000/test",
        "http://192.168.0.1:8080/test",
    ],
)
def test_validate_url(url: str):
    # Example usage
    assert validate_url(url)
