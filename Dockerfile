FROM python:3.9-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# China mirror for pip + transformer cache env
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_DEFAULT_TIMEOUT=200 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache \
    TORCH_HOME=/app/hf_cache \
    PYTHONUNBUFFERED=1

# Copy dependency list
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

# retry installation (important for slow China networks)
RUN pip install --timeout=200 --retries=10 -r requirements.txt

# Copy source code
COPY . .

# Create model/embedding cache
RUN mkdir -p /app/hf_cache

EXPOSE 7860

CMD ["python", "src/app.py"]
