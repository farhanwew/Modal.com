import requests

url = "https://farhanaryawicaksono--eros-mistral-modal-generate-endpoint.modal.run"
data = {
    "prompt": "Tell me about artificial intelligence",
    "max_tokens": 200,
    "temperature": 0.8
}

response = requests.post(url, json=data)
print(response.json())