FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copy and install dependencies -------------------------------------------------
# (requires a requirements.txt at the project root – see step 3 below)
COPY ml_platform/requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy the entire ml_platform package ------------------------------------------
COPY ml_platform/ ml_platform/

# Copy data & artifacts --------------------------------------------------------
COPY data/ data/
COPY artifacts/ artifacts/

EXPOSE 8000

CMD ["uvicorn", "ml_platform.api.main:app", "--host", "0.0.0.0", "--port", "8000"]