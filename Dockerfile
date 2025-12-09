# Use Python 3.10 for better PyTorch compatibility
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install python-dotenv for .env support
RUN pip install --no-cache-dir python-dotenv

# Copy application code
COPY main.py .
COPY api/ ./api/
COPY pipeline/ ./pipeline/
COPY models_manifest.json .

# Create models_cache directory (models will be downloaded at runtime from DigitalOcean)
RUN mkdir -p models_cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health')"

# Run the application
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]
