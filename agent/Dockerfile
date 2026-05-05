FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e ".[agent]"

CMD ["uvicorn", "agent.main:app", "--host", "0.0.0.0", "--port", "8001"]
