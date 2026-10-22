from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.abspath("../data/raw_logs")
os.makedirs(DATA_DIR, exist_ok=True)

class SessionPayload(BaseModel):
    participant_id: str
    trials: list

@app.post("/api/save-trial")
def save_trial(payload: SessionPayload):
    filename = f"{payload.participant_id}_{int(time.time())}.json"
    target_path = os.path.join(DATA_DIR, filename)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(payload.model_dump(), f, indent=2)
    return {"status": "success", "file": filename}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)