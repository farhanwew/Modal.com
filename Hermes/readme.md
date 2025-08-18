---

# 🔌 Request & cURL (Full Parameters)

## 1. cURL

```bash
curl -X POST "https://<your-app>.modal.run/generate-endpoint" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short story about a dragon guarding a mountain.",
    "system_prompt": "You are an epic narrator.",
    "prompt_template": "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    "max_tokens": 512,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 10,
    "min_p": 0.0,
    "typical_p": 1.0,
    "tfs": 1.0,
    "repeat_penalty": 1.1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "mirostat_mode": "Disabled",
    "mirostat_entropy": 5.0,
    "mirostat_learning_rate": 0.1,
    "seed": 42,
    "control_vector": "my_vector.npz"
  }'
```

---

## 2. Python (Requests)

```python
import requests

url = "https://<your-app>.modal.run/generate-endpoint"

payload = {
    "prompt": "Write a short story about a dragon guarding a mountain.",
    "system_prompt": "You are an epic narrator.",
    "prompt_template": "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    "max_tokens": 512,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 10,
    "min_p": 0.0,
    "typical_p": 1.0,
    "tfs": 1.0,
    "repeat_penalty": 1.1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "mirostat_mode": "Disabled",
    "mirostat_entropy": 5.0,
    "mirostat_learning_rate": 0.1,
    "seed": 42,
    "control_vector": "my_vector.npz"
}

res = requests.post(url, json=payload)
print(res.json())
```

---

## 3. Parameter Reference

| Parameter                | Type    | Default         | Description                                        |
| ------------------------ | ------- | --------------- | -------------------------------------------------- |
| `prompt`                 | `str`   | -               | **Required.** User input text                      |
| `system_prompt`          | `str`   | Default persona | Instruction for system role                        |
| `prompt_template`        | `str`   | Chat template   | Prompt formatting template                         |
| `max_tokens`             | `int`   | 512             | Max output tokens (1–4096)                         |
| `temperature`            | `float` | 0.8             | Sampling creativity (0.01–2.0)                     |
| `top_p`                  | `float` | 0.95            | Nucleus sampling (0.01–1.0)                        |
| `top_k`                  | `int`   | 10              | Top-k candidates (1–200)                           |
| `min_p`                  | `float` | 0.0             | Minimum probability cutoff                         |
| `typical_p`              | `float` | 1.0             | Typical sampling                                   |
| `tfs`                    | `float` | 1.0             | Tail free sampling                                 |
| `repeat_penalty`         | `float` | 1.1             | Repetition penalty (0.1–2.0)                       |
| `frequency_penalty`      | `float` | 0.0             | Penalize frequent tokens                           |
| `presence_penalty`       | `float` | 0.0             | Penalize already-seen tokens                       |
| `mirostat_mode`          | `str`   | Disabled        | `Disabled`, `1`, `2`, `mirostat_v1`, `mirostat_v2` |
| `mirostat_entropy`       | `float` | 5.0             | Tau parameter for Mirostat                         |
| `mirostat_learning_rate` | `float` | 0.1             | Eta parameter for Mirostat                         |
| `seed`                   | `int`   | None            | Random seed for deterministic output               |
| `control_vector`         | `str`   | None            | `.npz` control vector filename                     |

---