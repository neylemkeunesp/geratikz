from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys

def run(port=5000):
    server_address = ('127.0.0.1', port)
    print(f"Attempting to bind to 127.0.0.1:{port}...")
    try:
        httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
        print(f"Successfully bound to port {port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server")
            httpd.server_close()
            sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    run()
