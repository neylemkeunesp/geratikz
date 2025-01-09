from wsgiref.simple_server import make_server
import logging
import sys
import signal

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('wsgi.log')
    ]
)
logger = logging.getLogger(__name__)

def handle_signal(signum, frame):
    logger.info(f"Received signal {signum}")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def simple_app(environ, start_response):
    """Simplest possible WSGI application"""
    status = '200 OK'
    headers = [('Content-type', 'text/plain; charset=utf-8')]
    start_response(status, headers)

    return [b"Hello, World!"]

if __name__ == '__main__':
    try:
        logger.info("Starting WSGI server...")
        httpd = make_server('', 5000, simple_app)
        logger.info("Server running on port 5000...")
        httpd.serve_forever()
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
