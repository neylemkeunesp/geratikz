import asyncio
import logging
import sys
from fastapi import FastAPI
import uvicorn

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

async def main():
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="debug",
        access_log=True
    )
    server = uvicorn.Server(config)
    try:
        logger.info("Starting server...")
        await server.serve()
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
