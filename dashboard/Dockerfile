FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]