import socket
import sys
import logging
import select
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def test_socket():
    try:
        # Log system info
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Process ID: {os.getpid()}")
        
        # Create socket
        logger.info("Creating socket...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Set socket to non-blocking
        sock.setblocking(False)
        
        # Bind socket
        host = '127.0.0.1'
        port = 5000
        logger.info(f"Binding to {host}:{port}...")
        sock.bind((host, port))
        
        # Listen
        logger.info("Listening...")
        sock.listen(1)
        
        logger.info(f"Socket successfully bound and listening on {host}:{port}")
        
        # Wait for connection with timeout
        timeout = 10
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            try:
                ready = select.select([sock], [], [], 1.0)
                if ready[0]:
                    conn, addr = sock.accept()
                    logger.info(f"Connected by {addr}")
                    conn.send(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nSocket test successful!")
                    conn.close()
                    break
            except select.error as e:
                logger.error(f"Select error: {e}", exc_info=True)
                break
            except socket.error as e:
                logger.error(f"Socket error: {e}", exc_info=True)
                break
            
            logger.debug("Waiting for connection...")
        
        if (time.time() - start_time) >= timeout:
            logger.info("Timeout reached, no connection received")
            
    except Exception as e:
        logger.error(f"Socket error: {e}", exc_info=True)
    finally:
        logger.info("Closing socket...")
        sock.close()

if __name__ == "__main__":
    test_socket()
