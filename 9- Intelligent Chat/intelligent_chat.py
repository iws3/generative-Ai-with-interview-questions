from langchain.chains import RouterChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from research_chain import research_topic
from multimodal_qa import analyze_image_with_question

class IntelligentChat:
    def __init__(self):
        # Simple outing Logic - no complex langchain router needed for basic routing
        self.image_keywords= ['image', 'picture', 'photo', 'analyze this', 'whatis in this']
        self.research_keywords = ['research', 'what is ', 'who is', 'explain', 'tell me about']

        def route_query(self,query: str, has_image:bool=False)
        """Simple but effective routing logic"""

        # Priority 1: Image queries
        if has_image:
            return "image_analysis"
        # Prioroty 2: Research /Knowlwedge queries
        query_lower=query.lower()