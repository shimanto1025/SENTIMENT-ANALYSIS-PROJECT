import faiss
import numpy as np
import pickle
import json
from typing import List, Dict, Any
import os

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class SentimentVectorStore:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        self.metadata = []
        
        # Optional Redis integration
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host='vector-db',  # Docker service name
                    port=6379,
                    db=0,
                    decode_responses=True,
                    socket_connect_timeout=2
                )
                self.redis_client.ping()
                print("✅ Connected to Redis for metadata storage")
            except:
                print("⚠️  Redis not available, using in-memory storage")
                self.redis_client = None
    
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
        
        # Store in Redis if available, otherwise in memory
        if self.redis_client:
            for i, (text, sentiment) in enumerate(zip(texts, sentiments)):
                idx = self.index.ntotal - len(texts) + i
                self.redis_client.hset(f"text:{idx}", mapping={
                    "text": text,
                    "sentiment": json.dumps(sentiment),
                    "embedding_index": str(idx)
                })
        else:
            # Store texts and metadata in memory
            self.texts.extend(texts)
            self.metadata.extend(sentiments)
        
        print(f"✅ Added {len(texts)} texts to vector store")
    
    def search(self, query_embedding: np.ndarray, k: int = 5):
        """Search for similar texts"""
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype('float32')
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                if self.redis_client:
                    # Retrieve from Redis
                    data = self.redis_client.hgetall(f"text:{idx}")
                    if data:
                        results.append({
                            'text': data['text'],
                            'sentiment': json.loads(data['sentiment']),
                            'distance': float(distances[0][i])
                        })
                else:
                    # Retrieve from memory
                    if idx < len(self.texts):
                        results.append({
                            'text': self.texts[idx],
                            'sentiment': self.metadata[idx],
                            'distance': float(distances[0][i])
                        })
        
        return results
    
    def get_stats(self):
        """Get statistics about the vector store"""
        stats = {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'redis_connected': self.redis_client is not None
        }
        
        if self.redis_client:
            stats['redis_keys'] = len(self.redis_client.keys("text:*"))
        else:
            stats['texts_in_memory'] = len(self.texts)
            stats['metadata_in_memory'] = len(self.metadata)
        
        return stats
    
    def save(self, path: str):
        """Save vector store to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}_index.faiss")
        
        # Save texts and metadata (for in-memory storage)
        if not self.redis_client:
            with open(f"{path}_data.pkl", 'wb') as f:
                pickle.dump({'texts': self.texts, 'metadata': self.metadata}, f)
            print(f"✅ Vector store saved to {path} (in-memory mode)")
        else:
            # If using Redis, just save FAISS index
            print(f"✅ FAISS index saved to {path}_index.faiss (Redis mode)")
    
    def load(self, path: str):
        """Load vector store from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}_index.faiss")
        
        # Only load from pickle if not using Redis
        if not self.redis_client and os.path.exists(f"{path}_data.pkl"):
            with open(f"{path}_data.pkl", 'rb') as f:
                data = pickle.load(f)
                self.texts = data['texts']
                self.metadata = data['metadata']
            print(f"✅ Vector store loaded from {path} (in-memory mode)")
        else:
            print(f"✅ FAISS index loaded from {path}_index.faiss")
        
        return self