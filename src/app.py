import gradio as gr
import numpy as np
from models.sentiment_model import SentimentAnalyzer
from models.vector_store import SentimentVectorStore
from preprocessing.text_processor import TextPreprocessor
import json

# Initialize components
sentiment_analyzer = SentimentAnalyzer()
vector_store = SentimentVectorStore()
preprocessor = TextPreprocessor()

# Sample texts to populate vector store (in production, load from dataset)
sample_texts = [
    "I absolutely love this product! It's amazing.",
    "This is the worst experience I've ever had.",
    "The service was okay, nothing special.",
    "Highly recommended! Will buy again.",
    "Terrible quality, very disappointed.",
    "Good value for money.",
    "Excellent customer service, very helpful.",
    "Not worth the price at all."
]

def initialize_vector_store():
    """Initialize vector store with sample data"""
    embeddings = []
    sentiments = []
    
    for text in sample_texts:
        # Get sentiment
        sentiment = sentiment_analyzer.predict(text)
        sentiments.append(sentiment)
        
        # Get embedding
        embedding = sentiment_analyzer.get_embedding(text)
        embeddings.append(embedding)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    # Add to vector store
    vector_store.add_texts(sample_texts, embeddings_array, sentiments)
    
    return f"Vector store initialized with {len(sample_texts)} texts"

def analyze_sentiment(text):
    """Analyze sentiment of input text"""
    if not text.strip():
        return "Please enter some text", "", []
    
    # Preprocess text
    processed_text = preprocessor.preprocess(text)
    
    # Get sentiment prediction
    result = sentiment_analyzer.predict(text)
    
    # Get embedding and search for similar texts
    embedding = sentiment_analyzer.get_embedding(text)
    similar_texts = vector_store.search(embedding, k=3)
    
    # Format similar texts
    similar_formatted = ""
    for item in similar_texts:
        similar_formatted += f"Text: {item['text'][:100]}...\n"
        similar_formatted += f"Sentiment: {item['sentiment']['sentiment']} (Confidence: {item['sentiment']['score']})\n"
        similar_formatted += f"Similarity Distance: {item['distance']:.4f}\n"
        similar_formatted += "-" * 50 + "\n"
    
    return processed_text, result, similar_formatted

def format_sentiment_output(result):
    """Format sentiment result for display"""
    if isinstance(result, dict):
        sentiment_emoji = "😊" if result['sentiment'] == 'positive' else "😞"
        return f"""
        **Sentiment:** {sentiment_emoji} **{result['sentiment'].upper()}**
        
        **Confidence Score:** {result['score']:.2%}
        
        **Label:** {result['label']}
        """
    return result

# Initialize vector store
init_message = initialize_vector_store()
print(init_message)

# Create Gradio interface
with gr.Blocks(title="Sentiment Analysis with FAISS Vector Store", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎯 Sentiment Analysis System")
    gr.Markdown("Analyze text sentiment and find similar texts using FAISS vector database")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Enter text for sentiment analysis",
                placeholder="Type your text here...",
                lines=5
            )
            analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### System Status")
            status = gr.Textbox(
                label="Vector Store Status",
                value=init_message,
                interactive=False
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔧 Preprocessed Text")
            processed_output = gr.Textbox(label="Cleaned & Tokenized Text", interactive=False)
        
        with gr.Column():
            gr.Markdown("### 🎭 Sentiment Analysis Result")
            sentiment_output = gr.Markdown(label="Analysis Result")
    
    with gr.Row():
        gr.Markdown("### 🔍 Similar Texts from Vector Store")
        similar_output = gr.Textbox(
            label="Semantic Search Results",
            lines=8,
            interactive=False
        )
    
    # Example texts
    gr.Markdown("### 💡 Example Texts")
    examples = gr.Examples(
        examples=[
            ["I'm really happy with the service!"],
            ["This product is terrible and broke immediately."],
            ["The movie was average, not too good nor bad."],
            ["Exceptional quality and fast delivery!"]
        ],
        inputs=[input_text]
    )
    
    # Button actions
    analyze_btn.click(
        fn=analyze_sentiment,
        inputs=[input_text],
        outputs=[processed_output, sentiment_output, similar_output]
    ).then(
        fn=format_sentiment_output,
        inputs=[sentiment_output],
        outputs=[sentiment_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )