FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy root manifest and lock for dependency install
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra agent

# Copy the agent package code
COPY app/ ./app/
# Copy alembic for migrations
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]