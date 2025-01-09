import sys
import logging
import signal
from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fastapi.log')
    ]
)
logger = logging.getLogger(__name__)

# Signal handling
def handle_signal(signum, frame):
    logger.info(f"Received signal {signum}")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI application")
    yield
    logger.info("Shutting down FastAPI application")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    logger.info(f"Starting server with Python {sys.version}")
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=5000,
        log_level="debug",
        access_log=True,
        use_colors=False,
        workers=1
    )
    server = uvicorn.Server(config)
    try:
        server.run()
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
