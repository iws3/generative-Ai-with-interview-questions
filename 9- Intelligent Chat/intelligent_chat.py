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

    def route_query(self, query:str, has_image:bool=False):
         """Simple but effective routing logic"""

        # Priority 1: Image queries
        if has_image:
            return "image_analysis"
        # Prioroty 2: Research /Knowlwedge queries
        query_lower=query.lower()
        if any(keyword in query_lower for keyword in self.research_keywords):
            return "research"
        # Default: General LLM chat
        return "general_chat"
    
    def handle_chat(self, query:str, image_data:dict=None):
        """Main Chat handler with Router"""

        has_image = image_data is not None
        route = self.route_query(query, has_image)

        if route == "image_analysis":
            # Use your existing image analysis
            image = image_data['image'] # PIL Image Object
            answer = analyze_image_with_question(image, query)
            return {
                "query": query,
                "route": "image_analysis",
                "answer": answer,
                "tool": "Gemini Vision"
            }
        elif route == "research":
            # Use your research chain

            result = research_topic(query)
            return {
                "query": query,
                "route": "research",
                "answer": result.get("summary","Research failed"),
                "sentiment": result.get('sentiment','unknown'),
                "tool": "Wikipedia + Summarization"
            }
        
        else: #General chat
            # Use simple LLM for general conversation
            from transformers import pipeline
            chat_llm = pipeline('text-generation', model = 'distilgpt2', max_length=100)

            answer = chat_llm(query)[0]['generated_text']

            return {
                "query": query,
                "route": "general_chat",
                "answer": answer,
                "tool": "DistilGPT2"
            }

#  Initialize the intelligent chat system
chat_system = IntelligentChat()

def intelligent_chat(query:str, image_file =None):
    """Main function for the chat endpoint"""
    image_data = None
    if image_file:
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(image_file))
        image_data = { 'image': image}
    
    return chat_system.handle_chat(query, image_data)
    


    