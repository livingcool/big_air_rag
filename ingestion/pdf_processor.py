# ingestion/pdf_processor.py (IMPORTS REVISED)

import fitz # PyMuPDF
from typing import List, Dict, Any, Tuple
import os
from PIL import Image
from io import BytesIO
import traceback

# CRITICAL IMPORTS: Changed to relative imports (. means current package)
from .table_extractor import run_table_extraction 
  # .. means back one level (big_air_rag) then into embeddings
# .. # Relative import within the ingestion package is fine

# *** CRITICAL: Change these back to the full module path from the project root ***
from chunking.chunker import create_final_chunks  
from embeddings.embedder import index_chunks

# --- Configuration ---
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
DPI = 300 # Resolution for image rendering

# --- Helper Function for Coordinate Alignment ---

def is_text_in_cell(text_bbox: List[float], cell_bbox: List[float], tolerance: float = 0.9) -> bool:
    """
    Checks if a text block's bounding box is substantially contained within a cell's bounding box.
    """
    # 1. Convert to fitz.Rect for easy overlap calculation (PyMuPDF Rect handles [x0, y0, x1, y1])
    text_rect = fitz.Rect(text_bbox)
    cell_rect = fitz.Rect(cell_bbox)
    
    # 2. Calculate the intersection rectangle
    intersection_rect = text_rect.intersect(cell_rect)
    
    if intersection_rect.is_empty:
        return False
        
    overlap_area = intersection_rect.get_area()
    text_area = text_rect.get_area()
    
    if text_area == 0: 
        return False
        
    # Check if a substantial portion of the text block overlaps the cell
    return (overlap_area / text_area) >= tolerance


# ingestion/pdf_processor.py (Partial Update)
# ingestion/pdf_processor.py (SCALING FIX)
# ingestion/pdf_processor.py (AGGRESSIVE CAPTURE EDITION)

def align_text_to_structure(page_data: Dict[str, Any], table_structure: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merges PyMuPDF text blocks into the detected Table Transformer cell/row boxes.
    Includes PADDING and LOW TOLERANCE to ensure data is captured.
    """
    aligned_tables = []
    
    # Filter for the overall 'table' boundary boxes
    table_boxes = [e for e in table_structure if e['label'] == 'table']
    
    if not table_boxes:
        return []

    # --- SCALE FACTORS ---
    # Approx 72 / 300 = 0.24
    scale_x = 72 / DPI 
    scale_y = 72 / DPI 
    
    # Padding to ensure we catch edge cases (in points)
    PADDING = 15 

    for idx, table_box in enumerate(table_boxes):
        # 1. Get the Image-Space BBox (Pixels)
        img_bbox = table_box['box_2d'] 
        
        # 2. Convert to PDF-Space BBox (Points) and ADD PADDING
        table_rect_pdf = [
            (img_bbox[0] * scale_x) - PADDING, # Left
            (img_bbox[1] * scale_y) - PADDING, # Top
            (img_bbox[2] * scale_x) + PADDING, # Right
            (img_bbox[3] * scale_y) + PADDING  # Bottom
        ]
        
        table_output = {
            "table_id": f"t_{page_data['page_num']}_{idx}",
            "table_bbox": table_rect_pdf,
            "page_num": page_data['page_num'],
            "doc_id": page_data['doc_id'],
        }
        
        text_in_table = []
        # Iterate over raw text blocks from PyMuPDF
        for block in page_data.get("raw_pymupdf_output", {}).get("blocks", []):
            if block['type'] == 0: # Text block
                block_bbox = list(block['bbox'])
                
                # --- AGGRESSIVE FIX: Lower tolerance to 0.05 (5%) ---
                # If the text block overlaps the table box by even 5%, we take it.
                if is_text_in_cell(block_bbox, table_rect_pdf, tolerance=0.05): 
                    
                    # Extract text from SPANS inside lines
                    lines_text = []
                    for line in block.get('lines', []):
                        line_text = " ".join([span['text'] for span in line.get('spans', [])])
                        lines_text.append(line_text)
                    
                    block_text = " ".join(lines_text).strip()

                    if block_text:
                        text_in_table.append({
                            "text": block_text,
                            "bbox": block_bbox
                        })
        
        # Sort text blocks top-to-bottom, left-to-right
        text_in_table.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        # Final raw text content
        table_output['raw_text_content'] = " ".join([t['text'] for t in text_in_table]).strip()
        
        if table_output['raw_text_content']:
            aligned_tables.append(table_output)
        
    return aligned_tables
def extract_text_and_images(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Performs the initial pass: extracts text blocks and renders page images for VLMs.
    """
    doc_id = os.path.basename(pdf_path).split('.')[0]
    doc = fitz.open(pdf_path)
    page_data_list: List[Dict[str, Any]] = []

    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict", sort=True)
        img_path = None
        
        # Render Page to PNG Image
        try:
            pix = page.get_pixmap(dpi=DPI)
            img_data = pix.tobytes("png")
            img_filename = f"{doc_id}_page_{page_num+1}.png"
            img_path = os.path.join(PROCESSED_DIR, img_filename)
            img = Image.open(BytesIO(img_data))
            img.save(img_path)
        except Exception as e:
             print(f"  -> WARNING: Could not render image for Page {page_num + 1}: {e}")

        # Compile the output for the pipeline
        page_output = {
            "page_num": page_num + 1,
            "doc_id": doc_id,
            "image_path": img_path,
            "page_width": page.rect.width,
            "page_height": page.rect.height,
            "raw_pymupdf_output": page_dict,
            # --- CRITICAL FIX: Add this line to satisfy table_extractor.py ---
            "text_blocks": page_dict.get("blocks", []) 
        }
        page_data_list.append(page_output)

    doc.close()
    return page_data_list

def process_document_pipeline(file_path: str, doc_id: str) -> str:
    """
    Main orchestration function called by the FastAPI ingest endpoint.
    Runs the entire ingestion, extraction, and indexing process (Steps 1-5).
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    try:
        print(f"--- Starting Ingestion Pipeline for {doc_id} ---")

        # Step 1: Extract Text, Layout Data, and Page Images (PyMuPDF)
        page_data_list = extract_text_and_images(file_path)
        
        # Step 2: Run Table Structure Extraction (Table Transformer)
        table_detections_by_page = run_table_extraction(page_data_list) 
        
        # Step 3: Align Text to Structure (CRITICAL POST-PROCESSING)
        final_tables = []
        for page_data in page_data_list:
            page_detections = next((d for d in table_detections_by_page if d['page_num'] == page_data['page_num']), None)
            
            if page_detections and page_detections.get('detected_structure'):
                aligned_results = align_text_to_structure(page_data, page_detections['detected_structure'])
                final_tables.extend(aligned_results)

        # Step 4: Create Retrieval Chunks (Text, Tables, Figures)
        final_chunks = create_final_chunks(page_data_list, final_tables)

        # Step 5: Embeddings and Indexing (CRITICAL FINAL STEP)
        print("Starting Embedding and Indexing (Step 5/5)...")
        index_chunks(final_chunks) 

        print(f"--- Pipeline Execution Complete for {doc_id}. Indexed {len(final_chunks)} chunks. ---")
        
        return "SUCCESS"
        
    except Exception as e:
        print(f"FATAL INGESTION ERROR for {doc_id}: {e}")
        traceback.print_exc()
        return "FAILURE"


if __name__ == '__main__':
    # --- Local Test Stub ---
    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    TEST_PDF = "sample_imf_report.pdf" 
    test_path = os.path.join(RAW_DIR, TEST_PDF)

    if os.path.exists(test_path):
        # Ensure your table_extractor models are downloaded before running this
        result = process_document_pipeline(test_path, TEST_PDF.split('.')[0])
        print(f"\nLocal Test Result: {result}")
    else:
        print(f"ERROR: Place a sample PDF named '{TEST_PDF}' in the '{RAW_DIR}' directory to run the local test.")