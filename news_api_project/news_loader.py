"""
Module for loading news articles from the NewsData.io API.

This module defines a NewsLoader class that encapsulates the logic for
making API requests to NewsData.io, handling parameters, and returning
the JSON response containing news articles.
"""

import requests  # Used for making HTTP requests to external APIs.
from config import get_news_api_key  # Imports the function to retrieve the API key.

class NewsLoader:
    """
    A class to load news articles from the NewsData.io API.

    This class handles the construction of API requests and the retrieval
    of news data based on a provided API key and optional parameters.
    """
    def __init__(self, api_key: str):
        """
        Initializes the NewsLoader with the provided API key.

        Args:
            api_key (str): The API key for NewsData.io.

        How changes would affect the code:
        - Changing 'api_key' after initialization would require creating a new
          NewsLoader instance or directly modifying 'self.api_key'.
        - Modifying 'self.base_url' would change the API endpoint being called.
        """
        self.api_key = api_key  # Stores the API key for authentication.
        self.base_url = "https://newsdata.io/api/1/news"  # The base URL for the NewsData.io API.

    def load(self, country: str = "us", language: str = "en", category: str = "general") -> dict | None:
        """
        Loads news articles from the NewsData.io API.

        Args:
            country (str): The country to retrieve news from (e.g., "us", "gb"). Defaults to "us".
            language (str): The language of the news articles (e.g., "en", "fr"). Defaults to "en".
            category (str): The category of news (e.g., "business", "sports"). Defaults to "general".

        Returns:
            dict | None: A dictionary containing the news data if the request is successful,
                         otherwise None.

        How changes would affect the code:
        - Modifying default parameters will change the default news fetched.
        - Changes to 'self.base_url' or 'self.api_key' will directly impact API connectivity.
        - Altering the 'params' dictionary structure must align with NewsData.io API documentation.
        - Removing the try-except block would make the application crash on network errors.
        """
        # Construct the parameters for the API request.
        params = {
            "apikey": self.api_key,  # The API key for authentication.
            "country": country,      # Filter news by country.
            "language": language,    # Filter news by language.
            "category": category     # Filter news by category.
        }
        try:
            # Make the GET request to the NewsData.io API.
            response = requests.get(self.base_url, params=params)
            # Raise an HTTPError for bad responses (4xx or 5xx).
            response.raise_for_status()
            # Return the JSON response from the API.
            return response.json()
        except requests.exceptions.RequestException as e:
            # Catch any request-related exceptions (e.g., network issues, invalid URL).
            print(f"Error loading news: {e}")
            return None

if __name__ == "__main__":
    # This block is executed only when the script is run directly (not imported as a module).
    # It serves as a simple test to demonstrate the functionality of the NewsLoader class.
    api_key = get_news_api_key()
    if api_key:
        # Initialize the NewsLoader with the retrieved API key.
        news_loader = NewsLoader(api_key)
        # Attempt to load news articles.
        news_data = news_loader.load()
        if news_data:
            print("News loaded successfully:")
            # Print the entire news data. In a real application, you might process this further.
            print(news_data)
        else:
            print("Failed to load news. Check API key and network connection.")
    else:
        print("News API Key not found. Ensure it's set in your .env file.")
