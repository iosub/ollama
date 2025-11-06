import base64
import json
import pathlib
import requests
import sys

OLLAMA_HOST = "http://localhost:11434"
MODEL = "hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M"
MODEL="hf.co/unsloth/Qwen3-VL-2B-Instruct-1M-GGUF:Q4_K_M" 
    
IMAGE_PATH = pathlib.Path(r"C:\IA\1.png")  # update if the image lives elsewhere

PROMPT = "Extrae todos los datos de la imagen"
SYSTEM_PROMPT = (
    "Eres un asistente que analiza imágenes y SIEMPRE responde con JSON válido. "
    "Debes identificar todos los textos legibles, números, fechas y nombres propios. "
    "Responde usando SOLAMENTE el siguiente formato y sigue el ejemplo:\n"
    "{\n  \"descripcion_general\": <texto>,\n"
    "  \"elementos\": [\n    {\n      \"categoria\": <tipo>,\n"
    "      \"valor\": <texto extraído>,\n"
    "      \"coords\": [x1, y1, x2, y2] (si están disponibles)\n    }\n  ]\n}\n"
    "Ejemplo:\n"
    "{\n  \"descripcion_general\": \"Resumen breve de lo que aparece\",\n"
    "  \"elementos\": [\n    {\n      \"categoria\": \"texto\",\n"
    "      \"valor\": \"Factura 001\",\n"
    "      \"coords\": [12, 45, 210, 78]\n    }\n  ]\n}\n"
    "No añadas comentarios ni texto fuera del JSON."
)

def main() -> None:
    if not IMAGE_PATH.exists():
        print(f"Image not found: {IMAGE_PATH}", file=sys.stderr)
        sys.exit(1)

    with IMAGE_PATH.open("rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("ascii")

    payload = {
        "model": MODEL,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": PROMPT,
                "images": [image_b64],
            },
        ],
        "options": {
            # keep defaults here; tweak to experiment with penalties, stop sequences, etc.
            "num_predict": 512,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "seed": 42,
            "min_p": 0.05,
            "repeat_penalty": 1.02,
            "presence_penalty": 0.1,
            "repeat_last_n": 256,
        },
        "stream": False,
    }

    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=600)
    resp.raise_for_status()

    data = resp.json()
    print(json.dumps(data, indent=2))
    print("done:", data.get("done"))
    print("done_reason:", data.get("done_reason"))
    print("eval_count:", data.get("eval_count"))
    print("total_duration:", data.get("total_duration"))
    print("prompt_eval_duration:", data.get("prompt_eval_duration"))
    print("eval_duration:", data.get("eval_duration"))
    print("raw response:")
    print(data.get("message", {}).get("content", ""))

if __name__ == "__main__":
    main()