import sys
import logging
import signal
from contextlib import contextmanager
import uvicorn
from fastapi import FastAPI

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger(__name__)

# Signal handling
def handle_signal(signum, frame):
    logger.info(f"Received signal {signum}")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

@contextmanager
def exception_logging():
    try:
        yield
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    logger.info(f"Starting server with Python {sys.version}")
    logger.info(f"Current working directory: {sys.path}")
    
    with exception_logging():
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=5000,
            log_level="debug",
            access_log=True,
            use_colors=False
        )
        server = uvicorn.Server(config)
        server.run()
