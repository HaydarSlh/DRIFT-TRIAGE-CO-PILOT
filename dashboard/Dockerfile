FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e ".[dashboard]"

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
