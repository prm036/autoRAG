FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY config/ config/

# Install Python dependencies
RUN pip install --no-cache-dir ".[all]"

# Create data directories
RUN mkdir -p data/raw data/processed data/eval data/vector_store logs experiments/results

# Expose the API port
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
