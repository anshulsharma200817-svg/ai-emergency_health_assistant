FROM python:3.11-slim

LABEL maintainer="AI Emergency Health Detection System"
LABEL description="Educational Software Simulation - Synthetic Data ML Pipeline"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data directories (they'll be backed by a volume)
RUN mkdir -p data/raw data/interim data/processed \
             models/classical models/deep \
             experiments/results reports

# Default entrypoint runs the full pipeline:
#   1. Generate all synthetic data  (100K tabular + 20K temporal + 20K signals)
#   2. Train all classical ML models (KNN, LR, DT, RF)
#   3. Run the full 54-test suite
CMD ["bash", "-c", "\
  echo '' && \
  echo '============================================================' && \
  echo '  AI Emergency Health Detection — Full Pipeline' && \
  echo '  Educational Software Simulation (Synthetic Data Only)' && \
  echo '============================================================' && \
  echo '' && \
  echo '[1/3] Generating synthetic datasets...' && \
  python3 generate_data.py && \
  echo '' && \
  echo '[2/3] Training classical ML models...' && \
  python3 run_pipeline.py --classical-only --skip-generate && \
  echo '' && \
  echo '[3/3] Running the 54-test suite...' && \
  python3 -m pytest tests -q --tb=short && \
  echo '' && \
  echo '============================================================' && \
  echo '  PIPELINE COMPLETE — All artifacts saved to /app/models/' && \
  echo '============================================================' \
"]
