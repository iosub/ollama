import base64
import json
import requests
from pathlib import Path

image_path = Path(r"C:\ia\1.png")

with image_path.open("rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

payload = {
    "model": "hf.co/unsloth/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M",
    "messages": [
        {
            "role": "user",
            "content": "Describe la imagen adjunta.",
            "images": [image_b64],
        }
    ],
    "stream": False,
}

resp = requests.post(
    "http://127.0.0.1:11434/api/chat",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload),
    timeout=60,
)

resp.raise_for_status()
print(resp.json())
print(resp.text)