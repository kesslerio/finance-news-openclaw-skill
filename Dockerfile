FROM python:3.13-slim
WORKDIR /app

# Install build dependencies and libstdc++ for numpy
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir openbb openbb-yfinance

# Copy application code
COPY . .

# Local tailnet LLM routes. Scheduled runs use Kalliope Ornith; DS4 is an
# explicit manual override only.
# The bearer token is injected at runtime via `docker run -e KALLIOPE_SERVING_API_KEY`.
ENV KALLIOPE_SERVING_API_KEY=
ENV FINANCE_NEWS_QWEN_BASE_URL=http://100.124.155.99:4000/v1
ENV FINANCE_NEWS_QWEN_MODEL=ornith-1.5:35b-medium
ENV FINANCE_NEWS_DS4_BASE_URL=http://100.120.26.16:8888/v1
ENV FINANCE_NEWS_DS4_API_KEY=dsv4-local
ENV FINANCE_NEWS_DS4_MODEL=deepseek-v4-flash-0731
ENV FINANCE_NEWS_SUPPRESS_VENV_WARNING=1
ENV OPENBB_QUOTE_BIN=/app/scripts/openbb-quote

# Default command (override via docker run args)
CMD ["python3", "scripts/briefing.py"]
