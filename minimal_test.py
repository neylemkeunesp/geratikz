import os
import requests

api_key = "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"

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

with open('test_output.log', 'w') as f:
    try:
        f.write("Making request...\n")
        response = requests.post(
            "https://api.openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        f.write(f"Status: {response.status_code}\n")
        f.write(f"Response: {response.text}\n")
    except Exception as e:
        f.write(f"Error: {str(e)}\n")
