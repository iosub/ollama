import argparse
import base64
import json
from pathlib import Path

import requests


def load_image(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode()


def build_payload(model: str, prompt: str, image_path: Path, stream: bool) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [load_image(image_path)],
            }
        ],
        "stream": stream,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            # "seed": 42,
            # "stop": ["<|im_end|>"]
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(r"C:\ia\2.png"),
        help="Ruta al archivo de imagen que se enviará al modelo.",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        help="Archivo de texto con el prompt a enviar. Si se omite, se usa el valor por defecto.",
    )
    parser.add_argument(
        "--model",
        # default="hf.co/unsloth/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M",
        # default="qwen3-vl:8b-instruct",
        default="hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M",
        help="Nombre del modelo a invocar.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Activa la respuesta en streaming.",
    )
    args = parser.parse_args()

    default_prompt = "Describe la imagen adjunta."

    if args.prompt:
        prompt_text = args.prompt.read_text(encoding="utf-8")
    else:
        prompt_text = default_prompt

    payload = build_payload(args.model, prompt_text, args.image, args.stream)

    with requests.post(
        "http://127.0.0.1:11434/api/chat",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=60,
        stream=args.stream,
    ) as resp:
        resp.raise_for_status()

        if args.stream:
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                chunk = json.loads(line)
                print(chunk)
                if chunk.get("done"):
                    break
        else:
            print(resp.json())


if __name__ == "__main__":
    main()