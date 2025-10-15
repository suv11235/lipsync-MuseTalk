from fastapi import FastAPI
from .routes import router as api_router

app = FastAPI(title="MuseTalk Orchestrator (POC)")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

app.include_router(api_router)




