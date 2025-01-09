import os
from dotenv import load_dotenv
import requests
import json

# Set up logging to file
import logging
logging.basicConfig(
    filename='api_test.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load and verify .env file
logging.info(f"Current working directory: {os.getcwd()}")
logging.info("Checking .env file...")
env_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(env_path):
    logging.info(f".env file found at: {env_path}")
    with open(env_path, 'r') as f:
        env_contents = f.read()
    logging.info("Contents of .env file:")
    for line in env_contents.splitlines():
        if 'KEY' in line:
            key_name = line.split('=')[0]
            logging.info(f"{key_name}=[HIDDEN]")
        else:
            logging.info(line)
else:
    logging.error(".env file not found!")

logging.info("Loading environment variables...")
load_dotenv(verbose=True)

# Get API key
api_key = os.getenv('OPENROUTER_API_KEY')
logging.info(f"API Key found: {'Yes' if api_key else 'No'}")
if api_key:
    logging.info(f"API Key starts with: {api_key[:10]}...")
    logging.info(f"API Key length: {len(api_key)}")
    logging.info(f"API Key format valid: {'Yes' if api_key.startswith('sk-or-') else 'No'}")
else:
    logging.error("No API key found in environment variables")
    logging.info(f"Available environment variables: {[k for k in os.environ.keys() if k.lower().endswith('key')]}")

# Test API request
headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "http://localhost:3333",
    "Content-Type": "application/json"
}

payload = {
    "model": "anthropic/claude-3-haiku",
    "messages": [
        {"role": "user", "content": "Hello"}
    ]
}

logging.info("Making test request to OpenRouter API...")
logging.info(f"Headers (sanitized): {json.dumps({k: (v[:10] + '...' if k == 'Authorization' else v) for k, v in headers.items()}, indent=2)}")
logging.info(f"Payload: {json.dumps(payload, indent=2)}")

try:
    response = requests.post(
        "https://api.openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=10
    )
    
    logging.info(f"Response status code: {response.status_code}")
    logging.info(f"Response headers: {dict(response.headers)}")
    logging.info(f"Response body: {response.text}")
    
except requests.exceptions.RequestException as e:
    logging.error(f"Request error: {str(e)}")
except Exception as e:
    logging.error(f"Unexpected error: {str(e)}")
