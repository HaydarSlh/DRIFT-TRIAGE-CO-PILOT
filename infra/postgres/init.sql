-- Drift Triage Co-Pilot — initial schema
-- These tables are managed by Alembic migrations. This file only creates
-- the raw tables needed for the platform's own data (models, drift_reports)
-- that the agent service reads but doesn't own.

CREATE TABLE IF NOT EXISTS models (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL UNIQUE,
    version     TEXT NOT NULL,
    mlflow_run  TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);