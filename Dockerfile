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

ENV KIMI_API_KEY=
ENV KIMI_API_BASE_URL=https://api.kimi.com/coding/
ENV FINANCE_NEWS_KIMI_MODEL=k2p5
ENV MINIMAX_CODING_PLAN_API_KEY=

# Default command (override via docker run args)
CMD ["python3", "scripts/briefing.py"]
