import re


def validate_url(url: str) -> bool:
    """Created by GPT
    Args:
        url(str): Target url
    Returns:
        bool: Validation Result
    """
    # Updated URL validation regex pattern
    url_regex = re.compile(
        r"^(https?:\/\/)?"  # Optional HTTP or HTTPS
        r"((([a-zA-Z0-9\-_]+\.)+[a-zA-Z]{2,})|"  # Domain name (e.g., example.com)
        r"(localhost)|"  # Localhost
        r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}))"  # IPv4 address
        r"(:\d+)?"  # Optional port (e.g., :80)
        r"(\/[a-zA-Z0-9\-._~:\/?#[\]@!$&\'()*+,;=]*)?$"  # Optional path/query string
    )

    return re.match(url_regex, url)
