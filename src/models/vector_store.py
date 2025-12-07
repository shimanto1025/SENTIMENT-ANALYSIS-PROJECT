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
            raise ValueError(f"Number of texts ({len(texts)}) must equal number of embeddings ({len(embeddings)})")
        
        # Convert to numpy array if not already
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        # Ensure embeddings are 2D
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        elif embeddings.ndim == 2 and embeddings.shape[0] == 1:
            # Already 2D with batch size 1
            pass
        else:
            # Stack multiple embeddings
            embeddings = np.vstack(embeddings)
        
        # Check dimension
        if embeddings.shape[1] != self.dimension:
            print(f"⚠️  Warning: Embedding dimension {embeddings.shape[1]} doesn't match expected {self.dimension}")
            # Adjust FAISS index dimension if needed
            if embeddings.shape[0] > 0:
                self.dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store texts and metadata
        self.texts.extend(texts)
        self.metadata.extend(sentiments)
        
        print(f"✅ Added {len(texts)} texts to vector store")
    
    def search(self, query_embedding: np.ndarray, k: int = 5):
        """Search for similar texts"""
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype('float32')
        distances, indices = self.index.search(query_embedding, min(k, len(self.texts)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.texts):  # Valid index
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
        
        print(f"✅ Vector store saved to {path}")
    
    def load(self, path: str):
        """Load vector store from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}_index.faiss")
        
        # Load texts and metadata
        with open(f"{path}_data.pkl", 'rb') as f:
            data = pickle.load(f)
            self.texts = data['texts']
            self.metadata = data['metadata']
        
        print(f"✅ Vector store loaded from {path}")
        return self