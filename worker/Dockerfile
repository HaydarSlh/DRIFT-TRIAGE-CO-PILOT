FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e ".[worker]"

CMD ["python", "-m", "worker.main"]
