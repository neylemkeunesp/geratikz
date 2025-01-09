import requests
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('key_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def test_api_key(api_key):
    logging.info(f"Testing API key: {api_key[:10]}...")
    logging.info(f"Key length: {len(api_key)}")
    logging.info(f"Key format valid: {'Yes' if api_key.startswith('sk-or-') else 'No'}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        # First try the status endpoint
        logging.info("Testing status endpoint...")
        status_response = requests.get(
            "https://openrouter.ai/api/v1/status",
            headers=headers,
            timeout=10
        )
        logging.info(f"Status code: {status_response.status_code}")
        logging.info(f"Response: {status_response.text}")
        
        if status_response.status_code != 200:
            logging.error("Status endpoint returned non-200 response")
            return False
            
        # If status is good, try a simple chat completion
        logging.info("Testing chat completion...")
        chat_response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": "anthropic/claude-3-haiku",
                "messages": [{"role": "user", "content": "Say hello"}]
            },
            timeout=10
        )
        logging.info(f"Status code: {chat_response.status_code}")
        logging.info(f"Response: {chat_response.text}")
        
        return chat_response.status_code == 200
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    api_key = "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"
    success = test_api_key(api_key)
    logging.info(f"API key test {'succeeded' if success else 'failed'}")
    sys.exit(0 if success else 1)
