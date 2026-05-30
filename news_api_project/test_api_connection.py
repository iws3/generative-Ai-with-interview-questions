import requests
from config import get_news_api_key

def test_connection():
    api_key = get_news_api_key()
    if not api_key:
        print("API Key not found. Please check your .env file.")
        return

    base_url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": api_key,
        "country": "us",
        "language": "en",
        "category": "general"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print(f"API Connection Test: Status Code - {response.status_code}")
        print("Sample Response (first 200 chars):")
        print(response.text[:200])
    except requests.exceptions.RequestException as e:
        print(f"API Connection Test Failed: {e}")

if __name__ == "__main__":
    test_connection()
