import http.client
import json
import ssl

def write_log(message):
    with open('http_test.log', 'a') as f:
        f.write(message + '\n')

# Clear log file
open('http_test.log', 'w').close()

write_log("Starting HTTP client test...")

try:
    # Create SSL context
    context = ssl.create_default_context()
    write_log("Created SSL context")
    
    # Create connection
    write_log("Connecting to api.openrouter.ai...")
    conn = http.client.HTTPSConnection("api.openrouter.ai", context=context)
    write_log("Connection established")
    
    # Set headers
    headers = {
        'Authorization': 'Bearer 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',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'http://localhost:3333'
    }
    
    # Make request
    write_log("Making request to /api/v1/status...")
    conn.request("GET", "/api/v1/status", headers=headers)
    write_log("Request sent")
    
    # Get response
    response = conn.getresponse()
    write_log(f"Response status: {response.status} {response.reason}")
    
    # Read response data
    data = response.read()
    write_log(f"Response data: {data.decode()}")
    
    # Close connection
    conn.close()
    write_log("Connection closed")
    
except Exception as e:
    write_log(f"Error occurred: {str(e)}")
    import traceback
    write_log(f"Traceback: {traceback.format_exc()}")

write_log("Test complete")
