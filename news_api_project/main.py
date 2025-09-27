import json
from news_loader import NewsLoader
from ai_enhancer import AIEnhancer
from config import get_news_api_key

def main():
    api_key = get_news_api_key()
    if not api_key:
        print("News API Key not found. Please check your .env file.")
        return

    news_loader = NewsLoader(api_key)
    # Attempt to load news. This will likely fail due to the API key issue.
    news_data = news_loader.load()

    if news_data and news_data.get('results'):
        enhancer = AIEnhancer()
        enhanced_articles = []
        for article in news_data['results']:
            print(f"Enhancing article: {article.get('title', 'No Title')}")
            enhanced_article = enhancer.enhance_article(article)
            enhanced_articles.append(enhanced_article)

        output_data = {"enhanced_news": enhanced_articles}
        with open("enhanced_news.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print("Enhanced news saved to enhanced_news.json")
    else:
        print("No news data to process or API key issue persists.")
        # Create a dummy enhanced_news.json for demonstration if no real data
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
    main()
