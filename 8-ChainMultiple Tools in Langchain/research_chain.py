from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline  # FIXED IMPORT
from transformers import pipeline

# Initialize components
wiki = WikipediaAPIWrapper()

# LLM for summarization and sentiment - FIXED for T5
llm_pipeline = pipeline(
    "text2text-generation",  # T5 models use text2text-generation
    model="google/flan-t5-small",
    max_length=200
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)  # FIXED: Use the class, not module

# Step 1: Wikipedia Research
research_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Research this topic and provide key facts: {topic}"
)

# Step 2: Summarization  
summary_prompt = PromptTemplate(
    input_variables=["research"],
    template="Summarize this in 3 bullet points: {research}"
)

# Step 3: Sentiment Analysis
sentiment_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Analyze the sentiment of this text (positive/negative/neutral): {summary}"
)

# Create the chains
research_chain = LLMChain(llm=llm, prompt=research_prompt, output_key="research")
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary") 
sentiment_chain = LLMChain(llm=llm, prompt=sentiment_prompt, output_key="sentiment")

# Connect them sequentially
full_chain = SequentialChain(
    chains=[research_chain, summary_chain, sentiment_chain],
    input_variables=["topic"],
    output_variables=["research", "summary", "sentiment"]
)

def research_topic(topic: str):
    """Chain Wikipedia research + summarization + sentiment analysis"""
    try:
        # Get Wikipedia content first
        wiki_content = wiki.run(topic)
        
        if not wiki_content or "No good Wikipedia search result" in wiki_content:
            return {"error": f"No Wikipedia results found for '{topic}'"}
        
        # Run through the AI chain
        result = full_chain({"topic": wiki_content[:1000]})  # Limit length
        
        return {
            "topic": topic,
            "research_source": "Wikipedia",
            "summary": result["summary"],
            "sentiment": result["sentiment"],
            "full_chain": "Wikipedia → Summarize → Sentiment"
        }
        
    except Exception as e:
        return {"error": f"Research failed: {str(e)}"}