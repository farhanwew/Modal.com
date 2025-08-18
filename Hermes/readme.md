## 1. cURL

```bash
curl -X POST "https://<your-app>.modal.run/generate-endpoint" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tuliskan puisi tentang langit malam",
    "system_prompt": "Kamu adalah penyair bijak.",
    "max_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repeat_penalty": 1.1,
    "mirostat_mode": "Disabled",
    "seed": 42
  }'
```

---

## 2. Python (Requests)

```python
import requests

url = "https://<your-app>.modal.run/generate-endpoint"

payload = {
    "prompt": "Tuliskan puisi tentang langit malam",
    "system_prompt": "Kamu adalah penyair bijak.",
    "max_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repeat_penalty": 1.1,
    "mirostat_mode": "Disabled",
    "seed": 42
}

res = requests.post(url, json=payload)
print(res.json())
```

---
