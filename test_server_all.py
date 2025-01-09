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
    logger.info("Starting server...")
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",  # Listen on all interfaces
            port=5000,
            log_level="debug",
            access_log=True,
            workers=1
        )
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
