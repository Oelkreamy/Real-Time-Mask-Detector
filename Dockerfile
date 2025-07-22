# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy production requirements first (for better Docker layer caching)
COPY app/requirements-production.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p model/weights model/config

# Set Python path
ENV PYTHONPATH=/app

# Expose the port (Render will use PORT env variable)
EXPOSE $PORT

# Health check (use PORT env variable)
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# Command to run the production application (Render will set PORT env variable)
CMD uvicorn app.main_production:app --host 0.0.0.0 --port ${PORT:-10000}
