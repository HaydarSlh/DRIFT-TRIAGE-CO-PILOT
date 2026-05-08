# Drift Triage Co-Pilot

Autonomous ML drift detection and triage system powered by a LangGraph agent.

## Services

| Service    | Description                              | Port  |
|------------|------------------------------------------|-------|
| platform   | FastAPI — drift scoring & model registry | 8000  |
| agent      | LangGraph agent + webhook receiver       | 8001  |
| worker     | Queue consumer — DLQ & retry             | —     |
| dashboard  | Streamlit HIL dashboard                  | 8501  |
| postgres   | Persistent storage                       | 5432  |
| redis      | Task queue                               | 6379  |

## Quick Start

```bash
cp .env.example .env   #only once
python -m ml_platform.train
docker compose up --build


docker compose exec agent uv run alembic upgrade head  #only once
docker compose down
Remove-Item -Recurse -Force mlruns
docker compose up
docker compose exec platform python /app/scripts/final_register.py 
```




docker compose exec platform python /app/scripts/final_register.py 