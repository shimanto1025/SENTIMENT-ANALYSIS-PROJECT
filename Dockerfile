# Dockerfile - Optimized for slow networks
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set pip timeout and retry settings
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=5 \
    PIP_NO_CACHE_DIR=1

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU first (smaller)
RUN pip install --no-cache-dir \
    torch==2.0.0+cpu \
    torchvision==0.15.1+cpu \
    torchaudio==2.0.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies one by one (better for debugging)
RUN pip install --no-cache-dir transformers==4.30.0
RUN pip install --no-cache-dir sentence-transformers==2.2.0
RUN pip install --no-cache-dir faiss-cpu==1.7.0
RUN pip install --no-cache-dir gradio==3.0.0
RUN pip install --no-cache-dir pandas==1.5.0
RUN pip install --no-cache-dir numpy==1.24.0
RUN pip install --no-cache-dir scikit-learn==1.0.0
RUN pip install --no-cache-dir python-dotenv==1.0.0
RUN pip install --no-cache-dir nltk==3.8.0
RUN pip install --no-cache-dir redis==4.5.0

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Copy application code
COPY src/ ./src/

# Create volume for FAISS data
RUN mkdir -p /app/faiss_data

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "src/app.py"]