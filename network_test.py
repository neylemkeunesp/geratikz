import socket
import requests
import logging

logging.basicConfig(
    filename='network_test.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_dns_resolution(hostname):
    try:
        ip = socket.gethostbyname(hostname)
        logging.info(f"Successfully resolved {hostname} to {ip}")
        return True
    except socket.gaierror as e:
        logging.error(f"Failed to resolve {hostname}: {e}")
        return False

def test_https_connection(url):
    try:
        response = requests.get(url, timeout=5)
        logging.info(f"Successfully connected to {url}")
        logging.info(f"Status code: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to connect to {url}: {e}")
        return False

if __name__ == "__main__":
    hosts_to_test = [
        "api.openrouter.ai",
        "google.com",  # Control test
    ]
    
    logging.info("Starting network tests...")
    
    for host in hosts_to_test:
        logging.info(f"\nTesting {host}:")
        if test_dns_resolution(host):
            test_https_connection(f"https://{host}")
        else:
            logging.error(f"Skipping HTTPS test for {host} due to DNS resolution failure")
