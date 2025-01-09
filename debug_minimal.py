import sys
import logging
import uvicorn
from fastapi import FastAPI

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    logger.info("Python version: %s", sys.version)
    logger.info("Starting server...")
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="debug",
            access_log=True
        )
    except Exception as e:
        logger.error("Failed to start server: %s", str(e), exc_info=True)
        sys.exit(1)
