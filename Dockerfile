# big_air_rag/Dockerfile
# Use a lightweight Python base image suitable for ML
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies needed for PyMuPDF (optional, but good practice)
# and Tesseract (Critical for OCR)
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port (Gunicorn/Uvicorn default)
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]