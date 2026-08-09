# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY application ./application
COPY domain ./domain
COPY llama ./llama
COPY infrastructure ./infrastructure
COPY main.py logging_config.py ./

# Install dependencies using uv
RUN uv sync --frozen

# Run the application
CMD ["uv", "run", "python", "serve.py"]
