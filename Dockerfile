# Kyiv air-alert forecast: the same Python version the results were produced with.
FROM python:3.12.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# LightGBM needs the OpenMP runtime (on macOS that is `brew install libomp`).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md DECISIONS.md ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY data/raw ./data/raw
COPY reports ./reports

# Run as an unprivileged user; it may write only the download cache and reports.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p data/live \
    && chown -R app data/live reports
USER app

ENTRYPOINT ["python", "scripts/predict.py"]
