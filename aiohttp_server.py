from aiohttp import web
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

async def handle(request):
    return web.Response(text="Hello, World!")

async def init_app():
    app = web.Application()
    app.router.add_get('/', handle)
    return app

if __name__ == '__main__':
    logger.info("Starting aiohttp server...")
    try:
        app = init_app()
        web.run_app(app, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
