import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# UPDATED IMPORT: Use the new, dedicated package for HuggingFace embeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

COLLECTION_NAME = "ai_agent_pipeline_rag_collection"
PDF_PATH = "dummy_doc.pdf"

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([OPENWEATHER_API_KEY, QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY]):
    raise EnvironmentError("One or more essential API keys/URLs (OPENWEATHER, QDRANT, GROQ) are missing from the environment. Please check your .env file.")

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

print("Configuration loaded. Core LLM and Embeddings initialized.")