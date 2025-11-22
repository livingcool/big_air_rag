# 📘 Study Guide: Multi-Modal RAG Architecture & Logic

**Project:** Big AIR Lab — Multi-Modal Document Intelligence  
**Author:** Dr. Giva (Strategic AI Lead)

---

## 1. System Blueprint (High-Level Architecture)

This project implements a **Decoupled RAG Architecture**. Unlike monolithic apps, it separates the "Heavy Compute" (Backend) from the "User Interaction" (Frontend).

### The Data Flow Pipeline

```mermaid
graph TD
    User[User Uploads PDF] --> Streamlit[Frontend (Streamlit)]
    Streamlit -->|POST /ingest| API[Backend API (FastAPI)]
    
    subgraph "RAG Processing Engine (The Brain)"
        API --> Processor[PDF Processor]
        Processor -->|Text Stream (72 DPI)| PyMuPDF
        Processor -->|Image Stream (300 DPI)| TableTransformer[Vision Model]
        
        TableTransformer -->|BBox Coords| Alignment[Coordinate Scaler]
        PyMuPDF -->|Text Blocks| Alignment
        
        Alignment -->|Structured Chunks| Chunker
        Chunker -->|Text List| Embedder[SentenceTransformer]
        Embedder -->|Vectors (384-d)| Faiss[Vector DB]
    end
    
    User -->|Query| Streamlit
    Streamlit -->|POST /query| API
    API -->|Vector Search| Faiss
    Faiss -->|Top K Chunks| Reranker[MMR Algorithm]
    Reranker -->|Context| Gemini[Google Gemini 1.5 Flash]
    Gemini -->|Answer + Citations| User
```

---

## 2. Core Logic Breakdown

### A. Ingestion Layer (`ingestion/`)

**Goal:** Convert a PDF into clean, structured data, ensuring tables are captured as semantic units, not broken text.

#### The "Dual-Stream" Strategy

Standard OCR reads left-to-right, often destroying table rows. We use two streams:

1. **Visual Stream:** We render the PDF page as a high-res image (300 DPI) and send it to `microsoft/table-transformer`. This model "sees" the table and draws a bounding box around it.
2. **Text Stream:** We use PyMuPDF to extract raw text and its coordinates (PDF Points, usually 72 DPI).

#### The "Coordinate Scaling" Algorithm (Critical Logic)

Since the Visual Stream (300 DPI) and Text Stream (72 DPI) use different coordinate systems, we must align them.

$$
\text{Scale Factor} = \frac{\text{PDF DPI (72)}}{\text{Image Render DPI (300)}} \approx 0.24
$$

**The Aggressive Capture Fix:**

Because models aren't perfect, the predicted table box might be slightly too tight, cutting off numbers. We implemented:

- **Padding:** Add 15 points to the detected box edges.
- **Tolerance:** If a text block overlaps the box by even 5% (0.05), include it.

```python
# Logic Pseudocode
for table_box in detected_tables:
    # 1. Scale Visual Coords to PDF Coords
    pdf_box = table_box * (72 / 300)
    
    # 2. Expand Box (Aggressive Capture)
    pdf_box = expand(pdf_box, padding=15)
    
    # 3. Alignment Check
    if text_block.intersects(pdf_box, tolerance=0.05):
        add_to_table_chunk(text_block)
```

---

### B. Embedding & Indexing Layer (`embeddings/`)

**Goal:** Convert text chunks into mathematical vectors for search.

- **Model:** `all-MiniLM-L6-v2` (Local).
  - **Why this model?** It is small (80MB), fast (CPU-friendly), and produces 384-dimensional vectors that are highly effective for semantic search.

- **Database:** Faiss (Facebook AI Similarity Search).
  - **Index Type:** `IndexFlatL2`. This performs an exact Euclidean distance search (Brute Force).
  - **Logic:** For datasets < 100k chunks, brute force is faster and more accurate than approximate methods (IVF/HNSW).

---

### C. Retrieval Layer (`retrieval/`)

**Goal:** Find the most relevant information while avoiding redundancy.

#### The Search Algorithm

1. **Vector Search:** Convert User Query -> Vector. Find the nearest 50 chunks in Faiss space.

2. **MMR Reranking (Maximal Marginal Relevance):**
   - **Problem:** If the user asks about "Revenue," standard search might return 5 chunks that are all identical copies of the revenue paragraph.
   - **Solution (MMR):** We select chunks that are similar to the Query but dissimilar to Already Selected Chunks.

$$
\text{Score} = \lambda \cdot \text{Sim}(Query, Doc) - (1 - \lambda) \cdot \text{MaxSim}(Doc, \text{SelectedDocs})
$$

We set $\lambda = 0.7$ (70% focus on relevance, 30% focus on diversity).

---

### D. LLM & Synthesis Layer (`llm/`)

**Goal:** Generate a faithful answer with citations.

- **Engine:** Google Gemini 1.5 Flash.
  - **Why?** High context window (1M tokens), fast inference, and currently free tier availability.

#### Prompt Engineering Logic:

- We inject a **Strict System Instruction:** "You must cite sources using [DOC:ID | PAGE:X | CHUNK:Y]".
- We provide the context block **before** the user query to ground the model.

#### Fallback Logic:

If the API fails (Quota/Network), the system detects the error and switches to "Heuristic Mode," where it simply displays the top 3 retrieved text snippets instead of crashing.

---

## 3. Key Technical Decisions (The "Why")

| Decision | Rationale |
|----------|-----------|
| **Decoupled Backend** | The ML models (PyTorch/Transformers) are too heavy (>500MB) for serverless platforms like Vercel. A Docker container on Render solves this. |
| **Local Embeddings** | Relying on OpenAI Embeddings caused a "Quota Exceeded" crash. Local models cost $0 and run offline, ensuring the index is always buildable. |
| **Streamlit UI** | React/Next.js takes days to build. Streamlit takes hours and provides native support for data visualization (JSON/Dataframes). |
| **Table Transformer** | Standard OCR (Tesseract) reads tables line-by-line, mixing columns. A Vision Transformer "sees" the grid structure, preserving row integrity. |

---

## 4. Data Lifecycle

1. **Raw PDF:** Unstructured binary file.

2. **Ingestion:**
   - Text Blocks (Strings + Coordinates).
   - Table Objects (Rows/Cols + Coordinates).

3. **Chunking:**
   - Text Chunk: "Inflation rose by 2%..."
   - Table Chunk: "Year 2024 | GDP 2.5% | Inflation 1.2%" (Linearized).

4. **Embedding:**
   - `[0.12, -0.98, 0.05, ...]` (384 floats).

5. **Retrieval:**
   - User Query: "What is the 2024 GDP?"
   - Match: Table Chunk (Distance: 0.45).

6. **Synthesis:**
   - Gemini Input: "Context: Year 2024 | GDP 2.5%... Question: What is 2024 GDP?"
   - Gemini Output: "The 2024 GDP is 2.5% [Cite]."

---

## 5. Evaluation Metrics to Watch

When testing the system, focus on:

- **Hit Rate:** Does the correct table chunk appear in the "Evidence Used" column?
- **Precision:** Does the extracted number (e.g., 2.4%) match the PDF exactly?
- **Latency:** Total time from Query -> Answer (Target: < 5 seconds).
- **Noise:** Does the answer include irrelevant text from headers/footers? (MMR helps reduce this).