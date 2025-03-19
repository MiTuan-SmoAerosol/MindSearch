import json
import ipdb
import requests

# Define the backend URL
url = "http://localhost:8002/solve"
headers = {"Content-Type": "application/json"}


# Function to send a query to the backend and get the response
def get_response(query):
    # Prepare the input data
    data = {"inputs": query}

    # Send the request to the backend
    response = requests.post(url, headers=headers, data=json.dumps(data), timeout=6000, stream=True)

    # Process the streaming response
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\n"):
        if chunk:
            # ipdb.set_trace()
            decoded = chunk.decode("utf-8")
            if decoded == "\r":
                continue
            if decoded[:6] == "data: ":
                decoded = decoded[6:]
            elif decoded.startswith(": ping - "):
                continue
            # ipdb.set_trace()
            response_data = json.loads(decoded)
            agent_return = response_data["response"]
            ipdb.set_trace()
            node_name = response_data["current_node"]
            print(f"Node: {node_name}, Response: {agent_return['response']}")


# Example usage
if __name__ == "__main__":
    query = "What is the weather like today in New York?"
    get_response(query)
