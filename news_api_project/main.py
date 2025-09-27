"""
Main script for Exercise 1: News API Loader with AI Enhancement.

This script orchestrates the process of fetching news articles from NewsData.io,
enhancing them with AI-generated summaries, sentiment analysis, topic extraction,
and credibility assessments, and then saving the results to a JSON file.
"""

import json  # Used for working with JSON data (saving enhanced news).
from news_loader import NewsLoader  # Custom class for loading news from NewsData.io.
from ai_enhancer import AIEnhancer  # Custom class for AI-powered news article enhancements.
from config import get_news_api_key  # Function to retrieve the NewsData.io API key.

def main():
    """
    Main function to execute the news loading and AI enhancement pipeline.

    This function performs the following steps:
    1. Retrieves the NewsData.io API key.
    2. Initializes the NewsLoader to fetch news articles.
    3. Initializes the AIEnhancer to process and enhance articles.
    4. Loops through fetched articles, applies AI enhancements, and collects results.
    5. Saves the enhanced articles to an 'enhanced_news.json' file.
    6. If news fetching fails (e.g., due to API key issues), it generates a dummy
       'enhanced_news.json' for demonstration purposes.

    How changes would affect the code:
    - Changes in API key retrieval (get_news_api_key) or NewsLoader/AIEnhancer
      classes would directly impact the behavior of this pipeline.
    - Modifying the 'news_data.get('results')' check would change how articles
      are processed or if dummy data is generated.
    - Altering the output file name ('enhanced_news.json') would change where
      the results are saved.
    """
    # Retrieve the NewsData.io API key from environment variables.
    api_key = get_news_api_key()
    if not api_key:
        print("News API Key not found. Please check your .env file.")
        return  # Exit if no API key is found.

    # Initialize the NewsLoader with the retrieved API key.
    news_loader = NewsLoader(api_key)
    # Attempt to load news articles. This step is prone to failure if the API key
    # is invalid or if there are network issues, as observed previously.
    news_data = news_loader.load()

    # Check if news data was successfully loaded and contains results.
    if news_data and news_data.get('results'):
        enhancer = AIEnhancer()  # Initialize the AI enhancer.
        enhanced_articles = []   # List to store enhanced articles.
        for article in news_data['results']:
            print(f"Enhancing article: {article.get('title', 'No Title')}")
            # Enhance each article using the AIEnhancer.
            enhanced_article = enhancer.enhance_article(article)
            enhanced_articles.append(enhanced_article)

        # Prepare the output data structure.
        output_data = {"enhanced_news": enhanced_articles}
        # Save the enhanced news articles to a JSON file.
        with open("enhanced_news.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print("Enhanced news saved to enhanced_news.json")
    else:
        # This block is executed if no real news data could be fetched,
        # typically due to the NewsData.io API key issue.
        print("No news data to process or API key issue persists.")
        # Create a dummy enhanced_news.json for demonstration purposes.
        dummy_article = {
            "title": "Dummy News Article",
            "description": "This is a dummy description because real news could not be fetched.",
            "content": "This is dummy content. The API key for NewsData.io is likely invalid or has expired. Please check your NewsData.io API key and try again.",
            "ai_summary": "Dummy summary: API key issue.",
            "ai_sentiment": {"label": "neutral", "score": 0.0},
            "ai_topics": ["dummy", "api key", "error"],
            "ai_credibility_assessment": "Dummy assessment: Cannot verify due to API error."
        }
        output_data = {"enhanced_news": [dummy_article]}
        with open("enhanced_news.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print("Created dummy enhanced_news.json due to API key issue.")


if __name__ == "__main__":
    # This block ensures that the main() function is called only when the script
    # is executed directly (not when imported as a module).
    main()
