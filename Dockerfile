# Use slim Python image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system deps (needed for numpy, torch sometimes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of project
COPY . .

# Expose Render’s port
EXPOSE 10000

# Start Flask
CMD ["python", "app.py"]
