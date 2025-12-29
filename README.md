github:https://github.com/shimanto1025/SENTIMENT-ANALYSIS-PROJECT
Sentiment Analysis System with FAISS Vector Database

#Project Overview
A production-ready sentiment analysis system that combines DistilBERT transformer models with FAISS vector database technology for semantic similarity search. The system features a dual interface with Gradio web UI and FastAPI REST API, all containerized using Docker for easy deployment.
.................................................

#Prerequisites
1.Python 3.9 or higher
2.Docker and Docker Compose (for containerized deployment)
3.Git

Option 1: Docker Deployment:
1. git clone : https://github.com/shimanto1025/SENTIMENT-ANALYSIS-PROJECT
2. cd sentiment-analysis-project
3. python src/download_model.py
4. docker-compose up --build

#Access the application
# Web UI: http://localhost:7860
# API Docs: http://localhost:7860/docs
# Health Check: http://localhost:7860/health


Option 2: Virtual Environment Setup:
1. Create Virtual Environment: python -m venv venv
2. source venv/bin/activate
venv\Scripts\activate(windows)
3. pip install -r requirements.txt
4. python src/app.py

Core Capabilities:

Real-time Sentiment Analysis - Classifies text as Positive/Negative with confidence scores

Semantic Similarity Search - Finds similar texts using FAISS vector database

Dual Interface - Gradio web UI + FastAPI REST API

Containerized Deployment - Docker for reproducibility


# created by KAZI SHIMANTO HAQUE 


