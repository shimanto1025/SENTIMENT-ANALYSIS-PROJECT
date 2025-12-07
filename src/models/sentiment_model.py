from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize sentiment analysis model
        Using DistilBERT fine-tuned on SST-2 (Stanford Sentiment Treebank)
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Force CPU usage to avoid GPU compatibility issues
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()  
        
        # Create pipeline for easy inference
        self.classifier = pipeline(
            "sentiment-analysis", 
            model=self.model_name, 
            device=-1  # -1 means CPU
        )
        
        print(f"✅ Model loaded on {self.device}")
    
    def predict(self, text):
        """Predict sentiment and return label with confidence score"""
        result = self.classifier(text)[0]
        return {
            'label': result['label'],
            'score': round(result['score'], 4),
            'sentiment': 'positive' if result['label'] == 'POSITIVE' else 'negative'
        }
    
    def get_embedding(self, text):
        """Get embedding vector for the text - returns 2D array [1, 768]"""
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():

            outputs = self.model(**inputs, output_hidden_states=True)

            last_hidden_state = outputs.hidden_states[-1]
            
            # Use the [CLS] token embedding (first token)
            # Shape: [batch_size, hidden_size] -> [1, 768]
            cls_embedding = last_hidden_state[:, 0, :]
            
            # Convert to numpy
            embedding_np = cls_embedding.cpu().numpy()
            
        print(f"🔧 Embedding shape: {embedding_np.shape}")
        return embedding_np
    
    def batch_predict(self, texts):
        """Predict sentiment for multiple texts"""
        return self.classifier(texts)