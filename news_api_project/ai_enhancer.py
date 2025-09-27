# Import the Google Generative AI library
import google.generativeai as genai

class AIEnhancer:
    """
    A class to enhance news article descriptions using Google's Gemini AI model.
    It provides functionalities for summarizing, sentiment analysis, identifying key topics,
    and assessing credibility.
    """
    def __init__(self, api_key):
        """
        Initializes the AIEnhancer with a Google Gemini API key.

        Args:
            api_key (str): Your Google Gemini API key.
        """
        # Configure the generative AI model with the provided API key
        genai.configure(api_key=api_key)
        # Initialize the GenerativeModel with the specified model version
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')

    def enhance_article(self, article_description):
        """
        Enhances a given news article description by generating a summary,
        performing sentiment analysis, extracting key topics, and assessing credibility.

        Args:
            article_description (str): The description of the news article to enhance.

        Returns:
            str: A formatted string containing the AI summary, sentiment, key topics,
                 and credibility assessment, or an error message if an exception occurs.
        """
        # Construct the prompt for the Gemini AI model
        prompt = f"""
        Analyze the following news article description and provide the following:
        1.  **AI Summary (2 sentences max):**
        2.  **Sentiment Analysis (positive/negative/neutral):**
        3.  **Key Topics (3-5 topics, comma-separated):**
        4.  **Credibility Assessment (brief explanation):**

        Article Description:
        {article_description}
        """
        try:
            # Generate content using the configured Gemini model
            response = self.model.generate_content(prompt)
            # Return the generated text content
            return response.text
        except Exception as e:
            # Handle any exceptions that occur during the AI enhancement process
            return f"An error occurred during AI enhancement: {e}"