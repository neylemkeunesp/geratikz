import sys
import logging
import signal
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Signal handlers
def handle_exit(signum, frame):
    logger.info(f"Received signal {signum}")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

@app.get("/")
async def root():
    return JSONResponse({"message": "Hello World"})

@app.get("/health")
async def health():
    return JSONResponse({"status": "healthy"})

if __name__ == "__main__":
    try:
        logger.info("Starting server...")
        uvicorn.run(
            "modified_server:app",
            host="127.0.0.1",
            port=3333,
            log_level="debug",
            reload=False,
            workers=1
        )
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
