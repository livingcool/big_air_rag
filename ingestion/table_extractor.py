# ingestion/table_extractor.py

import torch
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from PIL import Image
import os
from typing import List, Dict, Any

# --- Configuration & Initialization ---
# Model checkpoint for Structure Recognition (detects rows, columns, headers, cells)
STRUCTURE_MODEL_ID = "microsoft/table-transformer-structure-recognition"

# Global model and processor instances for efficiency (avoids reloading on every API call)
try:
    # Ensure Tesseract is globally accessible or handle its path here if needed
    from pytesseract import image_to_string
    print("Table Extractor: Loading Hugging Face models...")
    
    # 1. Image Processor: Resizes and normalizes the image for the model
    structure_processor = DetrImageProcessor.from_pretrained(STRUCTURE_MODEL_ID)
    
    # 2. Structure Recognition Model
    structure_model = TableTransformerForObjectDetection.from_pretrained(STRUCTURE_MODEL_ID)
    
    # Set to evaluation mode and move to CPU (Faiss/Torch heavy, so we conserve GPU if present)
    structure_model.eval()
    DEVICE = "cpu"
    # NOTE: In a production container, you MUST utilize a GPU if available for speed.
    
    print(f"Table Extractor: Models loaded successfully on {DEVICE}.")

except ImportError:
    print("CRITICAL: pytesseract is not installed or Tesseract binary is missing. Table extraction will fail.")
    structure_processor = None
    structure_model = None
except Exception as e:
    print(f"CRITICAL: Failed to load Table Transformer models: {e}")
    structure_processor = None
    structure_model = None


def normalize_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    """
    Converts normalized (0-1000) bounding box coordinates back to original pixel values.
    Table Transformer outputs normalized coordinates (0-1000 range).
    """
    xmin, ymin, xmax, ymax = bbox
    xmin = int(xmin * width / 1000)
    ymin = int(ymin * height / 1000)
    xmax = int(xmax * width / 1000)
    ymax = int(ymax * height / 1000)
    return [xmin, ymin, xmax, ymax]


def get_table_structure(image_path: str, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Runs the Table Transformer model on a page image to detect table structure elements 
    (rows, columns, headers) and reconstructs the table.
    """
    if structure_model is None:
        return []

    print(f"  -> Running Table Transformer on {os.path.basename(image_path)}")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # 1. Preprocess the image
    encoding = structure_processor(image, return_tensors="pt")
    
    # 2. Run inference
    with torch.no_grad():
        outputs = structure_model(**encoding)

    # 3. Post-process the results
    target_sizes = [image.size[::-1]]
    # We use a threshold of 0.9 for high precision in the MVP
    results = structure_processor.post_process_object_detection(
        outputs, 
        threshold=0.9, 
        target_sizes=target_sizes
    )[0]
    
    detected_elements = []
    
    # 4. Iterate over detections and store the structured data
    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        # Convert bounding box from normalized to pixel values
        bbox_pixel = normalize_bbox(box.tolist(), width, height)
        
        detected_elements.append({
            "box_2d": bbox_pixel,  # xmin, ymin, xmax, ymax in pixels
            "label": structure_model.config.id2label[label.item()],
            "confidence": round(score.item(), 4),
        })

    # 
    
    # NOTE: The current step only extracts the coordinates of rows/columns/cells.
    # A complete table extraction pipeline (Step 5) requires complex post-processing:
    # 5. Cell Intersection: Calculate cell bounding boxes from row/column lines.
    # 6. OCR/Text Alignment: Align the text blocks from pdf_processor.py with the detected cells.
    # 7. Semantic Reconstruction: Assemble the final structured JSON (rows of {header: value})

    # For the 24-hour MVP, we stop at providing the raw structured detection output.
    # We pass this detection list forward for simplified text-alignment in the Chunking step.
    
    return detected_elements


def run_table_extraction(page_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Orchestrates table structure recognition for all page images.
    """
    processed_table_data = []
    
    for page_data in page_data_list:
        image_path = page_data.get("image_path")
        
        if not image_path or not os.path.exists(image_path):
            continue
            
        # Run the Hugging Face model
        detected_elements = get_table_structure(image_path, page_data)
        
        # Filter for tables and append structured output
        # Here we should also run the separate Table Detection model if we wanted full E2E, 
        # but the Structure Recognition model usually detects the table bounding box too.

        if detected_elements:
            processed_table_data.append({
                "page_num": page_data['page_num'],
                "doc_id": page_data['doc_id'],
                "detected_structure": detected_elements,
                "text_blocks": page_data['text_blocks'] # Include original text for alignment
            })

    print(f"\nTable Extractor: Processed {len(processed_table_data)} pages with structural detections.")
    return processed_table_data