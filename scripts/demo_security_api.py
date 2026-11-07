"""Demo security API script."""

import requests


def fetch(url: str):
    """Perform a GET request to the given URL and return parsed JSON.
    Args:
        url: The endpoint to fetch.
    Returns:
        The JSON-decoded response.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    # Example usage
    data = fetch("http://192.168.1.10/api/data")
    print(data)
