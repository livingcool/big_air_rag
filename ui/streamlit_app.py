# ui/streamlit_app.py

import streamlit as st
import requests
import json
import os
from typing import List, Dict, Any

# --- Configuration (FastAPI RAG Engine Host) ---
# NOTE: This assumes the FastAPI container is running on localhost:8000
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# --- UI Layout ---

st.set_page_config(layout="wide", page_title="Big AIR Lab — Multi-Modal RAG QA")
st.title("📚 Multi-Modal Document Intelligence (RAG-Based QA)")
st.caption("High-Accuracy System for Complex Documents (Tables, Text, Images)")

# --- API Interaction Functions ---

def ingest_document_api(uploaded_file):
    """Calls the FastAPI /ingest endpoint with the file."""
    url = f"{API_BASE_URL}/ingest"
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    
    with st.spinner("Ingesting, Extracting Tables, and Indexing... (This takes time for the ML models!)"):
        try:
            response = requests.post(url, files=files)
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Ingestion failed or API error: {e}")
            return None

def query_api(query: str):
    """Calls the FastAPI /query endpoint."""
    url = f"{API_BASE_URL}/query"
    params = {"query": query, "doc_ids": ["all"]}
    
    with st.spinner("Retrieving, Reranking, and Generating Answer..."):
        try:
            response = requests.post(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Query failed or API error: {e}")
            return None

# --- Main UI Blocks ---

# 1. Sidebar for Document Upload/Ingestion
with st.sidebar:
    st.header("1. Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])
    
    if uploaded_file and st.button("Start Ingestion & Indexing"):
        ingestion_result = ingest_document_api(uploaded_file)
        if ingestion_result:
            st.success(f"✅ Indexed {ingestion_result['doc_id']}. Ready for query.")
            st.json(ingestion_result)

# 2. Main Query/Answer Area
st.header("2. Ask a Question")
query = st.text_area("Enter your question about the indexed documents:", 
                     placeholder="e.g., What was the government's budget deficit in 2022?",
                     height=100)

if st.button("Submit Query", type="primary") and query:
    if API_BASE_URL == "http://localhost:8000":
        st.warning("Ensure the Docker container is running locally!")
    
    query_result = query_api(query)
    
    if query_result:
        col1, col2 = st.columns([2, 1])
        
        # Column 1: Answer and Citations
        with col1:
            st.subheader("🤖 Answer")
            st.markdown(query_result['answer'])
            
            # Extract and display unique citation sources
            if query_result['citations']:
                st.info("Citations Found:")
                st.markdown("\n".join(f"- {c}" for c in query_result['citations']))

        # Column 2: Retrieved Evidence Chunks
        with col2:
            st.subheader("🔍 Evidence Used (MMR Reranked)")
            for i, chunk in enumerate(query_result['retrieved_chunks']):
                with st.expander(f"Chunk {i+1} | Page {chunk['page']} | Modality: {chunk['modality']}"):
                    st.json({
                        "Chunk ID": chunk['id'].split('-')[0],
                        "Score Meta": "N/A (MMR)",
                        "Snippet": chunk['source_text_snippet'],
                        "BBox": chunk['bbox'] 
                    })