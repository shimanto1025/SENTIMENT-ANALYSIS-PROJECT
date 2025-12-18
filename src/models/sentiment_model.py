import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize Sentiment Analyzer with HuggingFace model
        Works offline and inside Chinese networks using hf-mirror.com
        """
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HF_HOME"] = "/app/hf_cache"
        os.environ["TRANSFORMERS_CACHE"] = "/app/hf_cache"
        os.environ["TORCH_HOME"] = "/app/hf_cache"

        self.model_name = model_name

        # Load tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # CPU only (safe for docker)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        # Pipeline for inference
        self.classifier = pipeline(
            task="sentiment-analysis",
            model=model_name,
            device=-1
        )

        print(f"✔ Sentiment model loaded using China mirror on {self.device}")

    def predict(self, text):
        """Predict sentiment and return structured output"""
        prediction = self.classifier(text)[0]
        return {
            "label": prediction["label"],
            "score": round(prediction["score"], 4),
            "sentiment": "positive" if prediction["label"].upper() == "POSITIVE" else "negative"
        }

    def get_embedding(self, text):
        """Return 768-dim embedding using [CLS] token"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            cls_embedding = last_hidden_state[:, 0, :]  # first token

            embedding_np = cls_embedding.cpu().numpy()

        print(f"Embedding shape: {embedding_np.shape}")
        return embedding_np

    def batch_predict(self, texts):
        """Predict multiple sentences"""
        return self.classifier(texts)

