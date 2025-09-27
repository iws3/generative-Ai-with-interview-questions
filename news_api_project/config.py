"""
Configuration module for loading API keys from environment variables.

This module provides a function to securely load the NewsData.io API key
from a .env file, promoting best practices for handling sensitive information.
"""

import os  # Provides functions for interacting with the operating system.
from dotenv import load_dotenv  # Used to load environment variables from a .env file.

def get_news_api_key() -> str | None:
    """
    Loads the NewsData.io API key from the .env file.

    This function first loads environment variables from a .env file
    (if present in the current directory or parent directories) and then
    retrieves the value associated with the "NEWS_API_KEY" variable.

    How changes would affect the code:
    - If the environment variable name changes, this function needs to be updated.
    - If the .env file is not present or the key is not set, this function
      will return None, leading to API connection failures.

    Returns:
        str | None: The NewsData.io API key if found, otherwise None.
    """
    load_dotenv()  # Load environment variables from .env file.
    return os.getenv("NEWS_API_KEY")  # Retrieve the API key.

if __name__ == "__main__":
    # This block is executed only when the script is run directly (not imported as a module).
    # It serves as a simple test to verify that the API key can be loaded correctly.
    api_key = get_news_api_key()
    if api_key:
        print(f"News API Key loaded successfully: {api_key[:5]}...") # Print first 5 chars for security.
    else:
        print("News API Key not found. Ensure it's set in your .env file.")
