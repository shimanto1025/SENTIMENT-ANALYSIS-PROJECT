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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Create pipeline for easy inference
        self.classifier = pipeline(
            "sentiment-analysis", 
            model=self.model, 
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # For embeddings (optional - if you want to store sentiment embeddings)
        self.embedding_model = pipeline(
            "feature-extraction",
            model=self.model,
            tokenizer=self.tokenizer
        )
    
    def predict(self, text):
        """Predict sentiment and return label with confidence score"""
        result = self.classifier(text)[0]
        return {
            'label': result['label'],
            'score': round(result['score'], 4),
            'sentiment': 'positive' if result['label'] == 'POSITIVE' else 'negative'
        }
    
    def get_embedding(self, text):
        """Get embedding vector for the text"""
        embedding = self.embedding_model(text, return_tensors="pt")
        return embedding[0][0].mean(dim=0).detach().numpy()
    
    def batch_predict(self, texts):
        """Predict sentiment for multiple texts"""
        return self.classifier(texts)