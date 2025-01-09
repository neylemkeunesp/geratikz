from fastapi import FastAPI
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"Hello": "World"}

if __name__ == "__main__":
    logger.info("Starting server on port 3333")
    uvicorn.run(app, host="0.0.0.0", port=3333, log_level="debug")
