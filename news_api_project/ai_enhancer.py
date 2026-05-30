"""
Module for AI-powered news article enhancement.

This module defines an AIEnhancer class that uses various Hugging Face
transformer pipelines to add AI-generated summaries, sentiment analysis,
topic extraction, and credibility assessments to news articles.
"""

from transformers import pipeline  # Used to easily load and use pre-trained models from Hugging Face.

class AIEnhancer:
    """
    A class that provides AI enhancement capabilities for text, including summarization,
    sentiment analysis, topic extraction, and credibility assessment.

    It leverages pre-trained models from the Hugging Face Transformers library.
    """
    def __init__(self):
        """
        Initializes the AIEnhancer by loading various pre-trained Hugging Face pipelines.

        This process can be time-consuming and requires an internet connection
        to download models if they are not already cached locally.

        How changes would affect the code:
        - Changing the 'model' parameter for any pipeline would load a different
          pre-trained model, potentially altering the quality and characteristics
          of the generated enhancements.
        - Adding or removing pipelines would change the available enhancement types.
        """
        # Load sentiment analysis pipeline: 'distilbert-base-uncased-finetuned-sst-2-english'
        # is a fine-tuned DistilBERT model for sentiment classification (positive/negative).
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        # Load summarization pipeline: 'facebook/bart-large-cnn' is a BART model
        # fine-tuned for abstractive summarization.
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        # Load a generic text generation pipeline: 'distilgpt2' is a smaller, faster
        # version of GPT-2, used here for basic topic extraction and credibility assessment.
        # This can be replaced with more specialized models or custom logic for better results.
        self.text_generator = pipeline("text-generation", model="distilgpt2")

    def get_summary(self, text: str) -> str:
        """
        Generates a concise summary of the provided text.

        Args:
            text (str): The input text to be summarized.

        Returns:
            str: A string containing the generated summary.

        How changes would affect the code:
        - Adjusting 'max_length' or 'min_length' directly controls the length of the summary.
        - Changing 'do_sample' to True would introduce variability in summaries,
          making them less deterministic but potentially more creative.
        - If the underlying 'summarizer' model is changed, the quality and style
          of the summaries will be different.
        """
        # Use the pre-loaded summarizer pipeline.
        # 'max_length' and 'min_length' define the acceptable range for the summary's token count.
        # 'do_sample=False' ensures that the summary generation is deterministic.
        summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
        # The pipeline returns a list of dictionaries, we extract the 'summary_text' from the first result.
        return summary[0]['summary_text']

    def get_sentiment(self, text: str) -> dict:
        """
        Analyzes the sentiment of the provided text.

        Args:
            text (str): The input text for sentiment analysis.

        Returns:
            dict: A dictionary containing the sentiment label (e.g., 'POSITIVE', 'NEGATIVE')
                  and a confidence score.

        How changes would affect the code:
        - If the underlying 'sentiment_analyzer' model is changed, the sentiment
          labels and scores might differ.
        - The output format is dictated by the Hugging Face pipeline;
          modifications would require custom parsing if a different format is desired.
        """
        # Use the pre-loaded sentiment analysis pipeline.
        # The pipeline returns a list of dictionaries, we extract the first result.
        sentiment = self.sentiment_analyzer(text)
        return sentiment[0]

    def get_topics(self, text: str) -> list:
        """
        Extracts key topics from the provided text using a text generation model.

        This is a simple implementation that relies on prompting a generic LLM.
        For more robust topic extraction, consider using specialized models
        (e.g., for keyword extraction) or more sophisticated NLP techniques.

        Args:
            text (str): The input text from which to extract topics.

        Returns:
            list: A list of strings, where each string is a key topic.

        How changes would affect the code:
        - Modifying the 'prompt' directly influences how the LLM interprets the request
          and thus the quality and format of the extracted topics.
        - Changing 'max_length' for the text generator might truncate the topic list.
        - The parsing logic (splitting by comma) is basic; if the LLM generates
          topics in a different format, this parsing will need adjustment.
        - Replacing 'self.text_generator' with a dedicated topic extraction model
          would significantly improve accuracy and reliability.
        """
        # Construct a prompt to guide the text generation model for topic extraction.
        prompt = f"Extract 3-5 key topics from the following news article: {text}\nTopics:"
        # Generate text based on the prompt.
        topics_raw = self.text_generator(prompt, max_length=100, num_return_sequences=1)
        # Extract the generated topics by removing the prompt and stripping whitespace.
        topics_str = topics_raw[0]['generated_text'].replace(prompt, "").strip()
        # Basic parsing: split the string by commas and clean up each topic.
        return [topic.strip() for topic in topics_str.split(',') if topic.strip()]

    def get_credibility_assessment(self, text: str) -> str:
        """
        Provides a brief credibility assessment for the given text using a text generation model.

        This is a simple implementation that relies on prompting a generic LLM.
        For a more robust assessment, specialized models or external knowledge
        bases would be required.

        Args:
            text (str): The input text to be assessed for credibility.

        Returns:
            str: A string containing the generated credibility assessment.

        How changes would affect the code:
        - Modifying the 'prompt' directly influences the LLM's output and the
          nature of the credibility assessment.
        - 'max_length' controls the length of the generated assessment.
        - The quality of the assessment is highly dependent on the capabilities
          of the 'text_generator' model and the specificity of the prompt.
        """
        # Construct a prompt to guide the text generation model for credibility assessment.
        prompt = f"Provide a brief credibility assessment for the following news article: {text}\nAssessment:"
        # Generate text based on the prompt.
        assessment_raw = self.text_generator(prompt, max_length=100, num_return_sequences=1)
        # Extract the generated assessment by removing the prompt and stripping whitespace.
        return assessment_raw[0]['generated_text'].replace(prompt, "").strip()

    def enhance_article(self, article: dict) -> dict:
        """
        Enhances a single news article dictionary with AI-generated insights.

        It extracts relevant text content from the article and then applies
        summarization, sentiment analysis, topic extraction, and credibility
        assessment.

        Args:
            article (dict): The original news article as a dictionary.
                            Expected to have 'content', 'description', or 'title' keys.

        Returns:
            dict: A new dictionary containing the original article data
                  plus the AI-generated enhancements.

        How changes would affect the code:
        - The order of content extraction ('content' or 'description' or 'title')
          determines which text is prioritized for enhancement.
        - If any of the enhancement methods (get_summary, get_sentiment, etc.)
          are modified, their output will directly impact the enhanced article.
        - The structure of the 'enhanced_article' dictionary (e.g., key names
          like 'ai_summary') can be customized here.
        - If new enhancement methods are added to AIEnhancer, they should be
          called and their results added to 'enhanced_article' here.
        """
        enhanced_article = article.copy()  # Create a copy to avoid modifying the original article.
        # Prioritize content, then description, then title for enhancement.
        content = article.get('content', '') or article.get('description', '') or article.get('title', '')

        if content:
            # Apply each AI enhancement if content is available.
            enhanced_article['ai_summary'] = self.get_summary(content)
            enhanced_article['ai_sentiment'] = self.get_sentiment(content)
            enhanced_article['ai_topics'] = self.get_topics(content)
            enhanced_article['ai_credibility_assessment'] = self.get_credibility_assessment(content)
        else:
            # Provide default values if no content is found for enhancement.
            enhanced_article['ai_summary'] = "No content to summarize."
            enhanced_article['ai_sentiment'] = {"label": "neutral", "score": 0.0}
            enhanced_article['ai_topics'] = []
            enhanced_article['ai_credibility_assessment'] = "No content to assess."

        return enhanced_article

if __name__ == "__main__":
    # This block is executed only when the script is run directly (not imported as a module).
    # It serves as a simple test to demonstrate the functionality of the AIEnhancer class.
    enhancer = AIEnhancer()
    # Define a sample news article for testing purposes.
    sample_article = {
        "title": "Sample News Article",
        "description": "This is a sample description of a news article.",
        "content": "This is the full content of a sample news article. It talks about various things, some positive and some negative. We want to extract a summary, sentiment, topics, and a credibility assessment from this text. The weather is great today, but the economy is struggling."
    }
    # Enhance the sample article.
    enhanced_article = enhancer.enhance_article(sample_article)
    print("Enhanced Article:")
    # Print the enhanced article, including all AI-generated insights.
    print(enhanced_article)
