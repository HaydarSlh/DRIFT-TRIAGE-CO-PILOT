from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Literal
from datetime import datetime

# ------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------
class PredictionRequest(BaseModel):
    """Body of POST /predict"""
    model_config = ConfigDict(populate_by_name=True)

    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    # Aliases for dotted CSV column names
    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(alias="nr.employed")

class PredictionResponse(BaseModel):
    """Returned by /predict"""
    probability: float
    prediction: int

# ------------------------------------------------------------
# Drift alert webhook (Platform → Agent)
# Contract: drift-alert-v1
# ------------------------------------------------------------
class DriftAlertWindow(BaseModel):
    start: str   # ISO 8601
    end: str     # ISO 8601
    num_predictions: int

class DriftAlertDetails(BaseModel):
    psi: Dict[str, float]
    chi2: Dict[str, float]
    output_drift: float

class DriftAlert(BaseModel):
    """Payload sent to agent at POST /webhooks/drift"""
    timestamp: str                                # ISO 8601
    event_id: str                                  # UUID
    model_id: str  
    severity: Literal["green", "yellow", "red"]
    previous_severity: Literal["green", "yellow", "red"]
    current_window: DriftAlertWindow
    drift_details: DriftAlertDetails

# ------------------------------------------------------------
# Promotion request (Agent → Platform)
# Contract: promotion-v1
# ------------------------------------------------------------
class PromotionRequest(BaseModel):
    """Received at POST /registry/promote"""
    action: Literal["promote", "rollback"]
    model_version: str = Field(..., description="e.g. '2'")
    investigation_id: str = Field(..., description="UUID")
    approved_by: str
    approval_timestamp: str = Field(..., description="ISO 8601 with Z")
    reason: str

class PromotionResponse(BaseModel):
    """Returned after successful promotion/rollback"""
    status: Literal["promoted", "rolled_back"]
    new_production_version: str
    mlflow_run_id: str

# ------------------------------------------------------------
# Rollback request (Agent → Platform)
# Contract: rollback-v1
# ------------------------------------------------------------
class RollbackRequest(BaseModel):
    """Received at POST /registry/rollback"""
    model_id: str
    target_version: str = Field(
        default="",
        description="Version to roll Production back to. Empty = go 1 version behind current Production.",
    )
    reason: str = ""

class RollbackResponse(BaseModel):
    """Returned after successful rollback"""
    model_id: str
    previous_version: str = Field(..., description="The Production version that was demoted (or '' if none).")
    new_production_version: str
    mlflow_run_id: str = ""