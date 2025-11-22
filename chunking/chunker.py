# chunking/chunker.py

from typing import List, Dict, Any
import uuid
import re

# --- Configuration ---
MAX_TEXT_CHUNK_TOKENS = 400
OVERLAP_TOKENS = 80 # Used for sliding window text chunking

# --- Chunking Helper Functions ---

def segment_text_by_semantic_overlap(text: str, max_tokens: int, overlap: int) -> List[str]:
    """
    Splits long text blocks using sentence boundaries and a sliding overlap window 
    to preserve semantic continuity. (MVP: Simple split by sentence/paragraph)
    """
    # For MVP, we'll use a simple, robust split by newline/sentence, 
    # and rejoin chunks up to the token limit.
    sentences = re.split(r'(?<=[.?!;])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence # Start new chunk
        else:
            current_chunk += (" " + sentence).strip()
    
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

# chunking/chunker.py (Partial Update)

def create_text_chunks(page_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processes raw text blocks from PyMuPDF, splitting long blocks into smaller, 
    semantically preserved chunks.
    """
    all_chunks = []
    
    for page in page_data_list:
        doc_id = page['doc_id']
        page_num = page['page_num']
        
        # Iterate over raw PyMuPDF blocks (type 0=text)
        # Note: We use the 'text_blocks' key we added earlier, or fall back to raw output
        blocks = page.get("text_blocks") or page.get("raw_pymupdf_output", {}).get("blocks", [])

        for block in blocks:
            if block['type'] != 0: continue # Skip images/other types here

            # --- CRITICAL FIX START ---
            # Extract text from SPANS inside lines
            lines_text = []
            for line in block.get('lines', []):
                # Join text from all spans in the line
                line_text = " ".join([span['text'] for span in line.get('spans', [])])
                lines_text.append(line_text)
            
            block_text = " ".join(lines_text).strip()
            # --- CRITICAL FIX END ---

            if not block_text:
                continue

            bbox = block['bbox']
            
            # Simple header detection heuristic
            structural_tokens = ""
            if len(block_text.split()) < 20 and block_text.isupper():
                 structural_tokens = f"Heading: {block_text}"
            
            # Split the block if it exceeds the maximum size
            text_segments = segment_text_by_semantic_overlap(
                block_text, 
                MAX_TEXT_CHUNK_TOKENS, 
                OVERLAP_TOKENS
            )
            
            for i, segment in enumerate(text_segments):
                chunk = {
                    "id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "page": page_num,
                    "modality": "text",
                    "source_text_snippet": segment,
                    "structural_tokens": structural_tokens, 
                    "bbox": bbox, 
                    "chunk_tokens": len(segment.split())
                }
                all_chunks.append(chunk)
                
    return all_chunks
def create_table_chunks(aligned_tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transforms aligned table data into retrieval-ready textual representations.
    This is the core of the high-accuracy data retrieval.
    """
    table_chunks = []
    
    for i, table_data in enumerate(aligned_tables):
        doc_id = table_data['doc_id']
        page_num = table_data['page_num']
        bbox = table_data['table_bbox']

        # 1. Create a machine-readable textual representation
        # NOTE: Using the raw_text_content from the alignment step as the main chunk text
        # In a final system, this would be a structured Markdown/CSV rendering.
        table_text = f"Table Content from Page {page_num}: {table_data['raw_text_content']}"
        
        # 2. Assign high-value structural tokens
        structural_tokens = f"Table_{i}: Financial Data, Tabular, Page {page_num}"
        
        table_chunks.append({
            "id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "page": page_num,
            "modality": "table",
            "source_text_snippet": table_text,
            "structural_tokens": structural_tokens,
            "bbox": bbox, 
            "chunk_tokens": len(table_text.split())
        })
        
    return table_chunks


def create_figure_chunks(page_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    (Placeholder) Creates chunks from image captions and VLM-derived numeric summaries.
    """
    figure_chunks = []
    # In this step, we would run the VLM model (SmolVLM) to get the 'numeric summary text'.
    
    # For MVP, we will rely on captions found near images (a heuristic approach).
    # Since we haven't implemented VLM yet, this returns an empty list.
    print("Figure chunking placeholder: VLM analysis for numeric summaries is pending.")
    
    return figure_chunks


def create_final_chunks(page_data_list: List[Dict[str, Any]], aligned_tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Orchestrates the creation and compilation of all retrieval chunks.
    """
    
    text_chunks = create_text_chunks(page_data_list)
    table_chunks = create_table_chunks(aligned_tables)
    figure_chunks = create_figure_chunks(page_data_list)
    
    final_chunks = text_chunks + table_chunks + figure_chunks
    
    print(f"\nChunking Complete:")
    print(f"  -> Text Chunks: {len(text_chunks)}")
    print(f"  -> Table Chunks: {len(table_chunks)}")
    print(f"  -> Total Chunks Ready for Embedding: {len(final_chunks)}")
    
    return final_chunks