FROM python:3.13-slim
WORKDIR /app

# Install build dependencies and libstdc++ for numpy
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default command (override via docker run args)
CMD ["python3", "scripts/briefing.py"]
