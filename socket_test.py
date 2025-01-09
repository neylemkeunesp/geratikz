import socket
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def test_socket():
    try:
        # Create socket
        logger.info("Creating socket...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind socket
        host = '127.0.0.1'
        port = 5000
        logger.info(f"Binding to {host}:{port}...")
        sock.bind((host, port))
        
        # Listen
        logger.info("Listening...")
        sock.listen(1)
        
        logger.info(f"Socket successfully bound and listening on {host}:{port}")
        
        # Accept one connection
        logger.info("Waiting for connection...")
        conn, addr = sock.accept()
        logger.info(f"Connected by {addr}")
        
        # Send response
        conn.send(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nSocket test successful!")
        conn.close()
        
    except Exception as e:
        logger.error(f"Socket error: {e}", exc_info=True)
    finally:
        logger.info("Closing socket...")
        sock.close()

if __name__ == "__main__":
    test_socket()
