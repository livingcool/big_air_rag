# big_air_rag/app.py (FINAL CORE UPDATE)

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import re # Added for post-processing
# app.py
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# --- CRITICAL: Add this line exactly ---
from typing import List, Dict, Any
# ...
from ingestion.pdf_processor import process_document_pipeline 
from retrieval.retriever import query_rag_system, format_context_for_llm 
from llm.prompt_manager import generate_answer_with_citations
# ...
# Load environment variables
load_dotenv()

# --- Critical RAG Imports ---
from ingestion.pdf_processor import process_document_pipeline # Ingestion/Indexing
from retrieval.retriever import query_rag_system, format_context_for_llm # Retrieval/MMR
from llm.prompt_manager import generate_answer_with_citations # LLM/QA

app = FastAPI(
    title="Multi-Modal RAG Processing Engine",
    description="Containerized API for high-accuracy document intelligence (Table/Chart extraction).",
    version="1.0.0"
)

# Pydantic model for the query response
class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    retrieved_chunks: List[Dict] # Metadata of the chunks used

# Pydantic model for the ingestion response
class IngestionResponse(BaseModel):
    doc_id: str
    status: str
    message: str

# ----------------- Core Endpoints -----------------

@app.post("/ingest", response_model=IngestionResponse, tags=["Ingestion"])
async def ingest_document(file: UploadFile = File(...)):
    """
    Triggers the end-to-end multi-modal ingestion pipeline for a PDF or DOCX file.
    Runs OCR, Table Transformer, and Embedding.
    """
    # ... (Keep file type validation the same) ...
    if file.content_type not in ["application/pdf"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF supported in MVP.")
    
    doc_id = os.path.basename(file.filename).split('.')[0]
    
    # 1. Save file to temporary location
    file_path = f"data/raw/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # 2. Run the full pipeline
    status = process_document_pipeline(file_path, doc_id) 

    if status != "SUCCESS":
        raise HTTPException(status_code=500, detail=f"Ingestion failed for {doc_id}.")

    return IngestionResponse(
        doc_id=doc_id,
        status=status,
        message=f"Document ingestion successful. Indexed {doc_id}. Ready for query."
    )

@app.post("/query", response_model=QueryResponse, tags=["Retrieval"])
async def answer_query(query: str, doc_ids: List[str] = ["all"]):
    """
    Accepts a user query, runs vector retrieval, reranking, and generates an LLM answer with citations.
    """
    # 1. Retrieval (ANN Search + MMR Rerank)
    retrieved_chunks = query_rag_system(query, doc_ids)
    
    if not retrieved_chunks:
        return QueryResponse(
            answer="No relevant information was retrieved from the document index.",
            citations=[],
            retrieved_chunks=[]
        )

    # 2. Context Formatting
    formatted_context = format_context_for_llm(retrieved_chunks)
    
    # 3. LLM Generation
    llm_result = generate_answer_with_citations(query, formatted_context)
    
    return QueryResponse(
        answer=llm_result['answer'],
        citations=llm_result['citations'],
        retrieved_chunks=retrieved_chunks
    )

# ----------------- Run Uvicorn -----------------
# ... (Keep the uvicorn run block the same) ...