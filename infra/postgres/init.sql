-- Drift Triage Co-Pilot — initial schema

CREATE TABLE IF NOT EXISTS models (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL UNIQUE,
    version     TEXT NOT NULL,
    mlflow_run  TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS drift_reports (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id    UUID REFERENCES models(id),
    psi_score   FLOAT NOT NULL,
    chi2_score  FLOAT,
    decision    TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS triage_actions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id   UUID REFERENCES drift_reports(id),
    action      TEXT NOT NULL,
    task_id     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS comms_log (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id   UUID REFERENCES drift_reports(id),
    channel     TEXT NOT NULL,
    message     TEXT,
    sent_at     TIMESTAMPTZ DEFAULT now()
);
