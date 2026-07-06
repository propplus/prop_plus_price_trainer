FROM python:3.11-slim

WORKDIR /app

# System libs required by lightgbm
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "-m", "src.train"]
