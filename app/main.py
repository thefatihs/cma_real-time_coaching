from fastapi import FastAPI

app = FastAPI(
    title="CallMetric Live ASR",
    description="Real-time speech-to-text service for call center audio.",
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "CallMetric Live ASR is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}
