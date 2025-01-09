import http.server
import socketserver
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Server is running!")

if __name__ == "__main__":
    PORT = 5000
    Handler.protocol_version = "HTTP/1.0"
    
    logger.info(f"Starting basic HTTP server on port {PORT}")
    try:
        with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
            logger.info(f"Server running at http://127.0.0.1:{PORT}")
            httpd.serve_forever()
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
