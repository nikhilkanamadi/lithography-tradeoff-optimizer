FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || pip install --no-cache-dir .

# Application code
COPY . .

# Create model directory
RUN mkdir -p models

EXPOSE 8000

CMD ["uvicorn", "lto.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
