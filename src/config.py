import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Model Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
    
    # Vector Store Configuration
    VECTOR_STORE_DIMENSION = 768
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    
    # Application Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 7860))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # FAISS Configuration
    FAISS_INDEX_TYPE = "FlatL2"
    
    # Hugging Face Token
    HF_TOKEN = os.getenv("HF_TOKEN", "")