from fastapi import FastAPI, Request
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dummy_agent")

app = FastAPI()

@app.post("/webhooks/drift")
async def drift_webhook(request: Request):
    # Read and log the payload
    body = await request.json()
    logger.info("📩 Webhook received:")
    logger.info(f"  timestamp: {body.get('timestamp')}")
    logger.info(f"  event_id : {body.get('event_id')}")
    logger.info(f"  severity : {body.get('severity')}")
    logger.info(f"  previous : {body.get('previous_severity')}")
    logger.info(f"  psi keys : {list(body.get('drift_details', {}).get('psi', {}).keys())}")
    logger.info(f"  chi2 keys: {list(body.get('drift_details', {}).get('chi2', {}).keys())}")
    logger.info(f"  output_drift: {body.get('drift_details', {}).get('output_drift')}")

    # Respond exactly as the contract demands
    return {
        "investigation_id": "dummy-inv-0001"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)