FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/upload data/train data/test models logs static/css static/js templates

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=src/api.py
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "src/api.py"]