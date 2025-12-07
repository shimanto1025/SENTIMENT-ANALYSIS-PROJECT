import gradio as gr
import numpy as np
import os
from models.sentiment_model import SentimentAnalyzer
from models.vector_store import SentimentVectorStore
from preprocessing.text_processor import TextPreprocessor
import json

# Disable GPU to avoid compatibility issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Initialize components
sentiment_analyzer = SentimentAnalyzer()
vector_store = SentimentVectorStore()
preprocessor = TextPreprocessor()

# Sample texts to populate vector store
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
    print("Initializing vector store...")
    
    embeddings = []
    sentiments = []
    
    for i, text in enumerate(sample_texts):
        print(f"Processing text {i+1}/{len(sample_texts)}: '{text[:30]}...'")
        
        # Get sentiment
        sentiment = sentiment_analyzer.predict(text)
        sentiments.append(sentiment)
        
        # Get embedding
        embedding = sentiment_analyzer.get_embedding(text)
        embeddings.append(embedding)
    
    # Stack all embeddings into a single 2D array
    if len(embeddings) > 0:
        embeddings_array = np.vstack(embeddings)
    else:
        embeddings_array = np.array([])
    
    print(f"\n📊 Total embeddings shape: {embeddings_array.shape}")
    print(f"📝 Number of texts: {len(sample_texts)}")
    
    # Add to vector store
    vector_store.add_texts(sample_texts, embeddings_array, sentiments)
    
    return f"✅ Vector store initialized with {len(sample_texts)} texts"

def analyze_sentiment(text):
    """Analyze sentiment of input text"""
    if not text.strip():
        return "Please enter some text", "", []
    
    print(f"\n🔍 Analyzing: '{text[:50]}...'")
    
    # Preprocess text
    processed_text = preprocessor.preprocess(text)
    print(f"🧹 Preprocessed: '{processed_text[:50]}...'")
    
    # Get sentiment prediction
    result = sentiment_analyzer.predict(text)
    print(f"🎭 Sentiment: {result}")
    
    # Get embedding and search for similar texts
    embedding = sentiment_analyzer.get_embedding(text)
    similar_texts = vector_store.search(embedding, k=3)
    print(f"🔎 Found {len(similar_texts)} similar texts")
    
    # Format similar texts
    similar_formatted = ""
    for i, item in enumerate(similar_texts):
        similar_formatted += f"Result {i+1}:\n"
        similar_formatted += f"  📝 Text: {item['text'][:80]}...\n"
        similar_formatted += f"  🎭 Sentiment: {item['sentiment']['sentiment']} "
        similar_formatted += f"(Confidence: {item['sentiment']['score']:.2%})\n"
        similar_formatted += f"  📏 Similarity Distance: {item['distance']:.4f}\n"
        similar_formatted += "-" * 60 + "\n"
    
    # Format sentiment output
    sentiment_emoji = "😊" if result['sentiment'] == 'positive' else "😞"
    sentiment_color = "green" if result['sentiment'] == 'positive' else "red"
    
    sentiment_html = f"""
    <div style='padding: 20px; border-radius: 10px; background-color: #f5f5f5;'>
        <h3 style='color: {sentiment_color};'>Sentiment: {sentiment_emoji} {result['sentiment'].upper()}</h3>
        <p><strong>Confidence Score:</strong> <span style='font-size: 1.2em;'>{result['score']:.2%}</span></p>
        <p><strong>Label:</strong> {result['label']}</p>
    </div>
    """
    
    return processed_text, sentiment_html, similar_formatted

# Initialize vector store
print("=" * 50)
print("🚀 Starting Sentiment Analysis Application")
print("=" * 50)

init_message = initialize_vector_store()
print(init_message)

# FIXED: Correct theme syntax
with gr.Blocks(title="Sentiment Analysis with FAISS Vector Store") as demo:
    gr.Markdown("# 🎯 Sentiment Analysis System")
    gr.Markdown("Analyze text sentiment and find similar texts using FAISS vector database")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="📝 Enter text for sentiment analysis",
                placeholder="Type your text here...",
                lines=4
            )
            analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 System Status")
            status = gr.Textbox(
                label="Vector Store Status",
                value=init_message,
                interactive=False
            )
            clear_btn = gr.Button("🔄 Clear All", variant="secondary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔧 Preprocessed Text")
            processed_output = gr.Textbox(
                label="Cleaned & Tokenized Text", 
                interactive=False,
                lines=3
            )
        
        with gr.Column():
            gr.Markdown("### 🎭 Sentiment Analysis Result")
            sentiment_output = gr.HTML(label="Analysis Result")
    
    with gr.Row():
        gr.Markdown("### 🔍 Similar Texts from Vector Store")
        similar_output = gr.Textbox(
            label="Semantic Search Results",
            lines=6,
            interactive=False
        )
    
    # Example texts
    gr.Markdown("### 💡 Try These Examples")
    examples = gr.Examples(
        examples=[
            ["I'm really happy with the service!"],
            ["This product is terrible and broke immediately."],
            ["The movie was average, not too good nor bad."],
            ["Exceptional quality and fast delivery!"],
            ["I would not recommend this to anyone."]
        ],
        inputs=[input_text],
        label="Click any example to try it"
    )
    
    # Button actions
    analyze_btn.click(
        fn=analyze_sentiment,
        inputs=[input_text],
        outputs=[processed_output, sentiment_output, similar_output]
    )
    
    # Clear button action
    def clear_all():
        return "", "", ""
    
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[input_text, processed_output, similar_output]
    )

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🌐 Starting Gradio interface...")
    print("📱 Open http://localhost:7860 in your browser")
    print("=" * 50 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )