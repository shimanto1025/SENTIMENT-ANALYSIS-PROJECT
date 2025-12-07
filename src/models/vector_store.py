import faiss
import numpy as np
import pickle
from typing import List, Dict, Any
import os

class SentimentVectorStore:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        self.metadata = []  # Store sentiment info with each text
    
    def add_texts(self, texts: List[str], embeddings: np.ndarray, sentiments: List[Dict]):
        """Add texts and their embeddings to the vector store"""
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must equal number of embeddings")
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store texts and metadata
        self.texts.extend(texts)
        self.metadata.extend(sentiments)
    
    def search(self, query_embedding: np.ndarray, k: int = 5):
        """Search for similar texts"""
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):  # Valid index
                results.append({
                    'text': self.texts[idx],
                    'sentiment': self.metadata[idx],
                    'distance': float(distances[0][i])
                })
        
        return results
    
    def save(self, path: str):
        """Save vector store to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}_index.faiss")
        
        # Save texts and metadata
        with open(f"{path}_data.pkl", 'wb') as f:
            pickle.dump({'texts': self.texts, 'metadata': self.metadata}, f)
    
    def load(self, path: str):
        """Load vector store from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}_index.faiss")
        
        # Load texts and metadata
        with open(f"{path}_data.pkl", 'rb') as f:
            data = pickle.load(f)
            self.texts = data['texts']
            self.metadata = data['metadata']
        
        return self