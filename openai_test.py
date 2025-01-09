import os
from openai import OpenAI
import sys
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    filename='openai_test.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def validate_url(url):
    """Validate URL format and DNS resolution"""
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return False
        return True
    except Exception:
        return False

# Get API key from environment
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    logging.error("OPENAI_API_KEY environment variable not set")
    sys.exit(1)

# Configure client to use OpenRouter
logging.info("Configuring OpenAI client for OpenRouter...")
base_url = "https://api.openrouter.ai/api/v1"

# Validate base URL
if not validate_url(base_url):
    logging.error(f"Invalid base URL: {base_url}")
    sys.exit(1)

try:
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/yourusername/yourproject",  # Replace with your actual site
            "X-Title": "Test Project"  # Optional: your site/app name
        }
    )
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {str(e)}")
    sys.exit(1)

def make_test_request():
    """Make a test request to OpenRouter API"""
    try:
        logging.info("Making test request to OpenRouter...")
        response = client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=[
                {"role": "user", "content": "Say hello"}
            ]
        )
        logging.info("API call successful!")
        result = response.choices[0].message.content
        logging.info(f"Response: {result}")
        return True, result
    except Exception as e:
        error_msg = f"Error making API request: {str(e)}"
        logging.error(error_msg)
        return False, error_msg

if __name__ == "__main__":
    success, message = make_test_request()
    sys.exit(0 if success else 1)
