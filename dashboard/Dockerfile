FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    --timeout 300 \
    --retries 10 \
    -r requirements.txt

COPY . .

# SQLite store for dashboard accounts
RUN mkdir -p /app/data

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
