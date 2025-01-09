from http.server import HTTPServer, BaseHTTPRequestHandler
import logging
import sys
import socket

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Test server running!")
        logger.info(f"Handled request from {self.client_address}")

def run_server(host='0.0.0.0', port=5000):
    try:
        # Log system info
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Socket names: {socket.gethostname()}, {socket.getfqdn()}")
        logger.info(f"Attempting to bind to {host}:{port}")
        
        # Create server
        server = HTTPServer((host, port), SimpleHandler)
        logger.info(f"Server created and bound to {host}:{port}")
        
        # Start serving
        logger.info("Starting server...")
        server.serve_forever()
        
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    run_server()
