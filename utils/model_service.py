import ollama
import requests
import re
import os
from dotenv import load_dotenv

load_dotenv()
token = os.environ.get("OPENWEBUI_API_KEY")


def chat_with_model(model="llama4:latest", messages=None):
    url = "http://chat.ese.ic.ac.uk:8080/api/chat/completions"
    headers = {"Authorization": f"Bearer {token}", "Content-type": "application/json"}
    data = {
        "model": model,
        "messages": messages,
    }
    response = requests.post(url, headers=headers, json=data)
    raw = response.json()
    return raw["choices"][0]["message"]["content"]


def chat_with_model_thinking(model="deepseek-r1:70b", messages=None):
    res = chat_with_model(model, messages)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", res, flags=re.DOTALL)
    return cleaned.strip()


# deepseek-r1:1.5b
# qwen3:1.7b
# llama3.2:3b
# qwen3:8b
def chat_with_ollama_local(model="qwen3:1.7b", messages=None, temperature=0.1):
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature, "top_p": 0.9, "num_predict": 1024},
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"Ollama local call failed: {e}")
        raise Exception(f"Ollama call failed: {e}")


if __name__ == "__main__":
    res = chat_with_ollama_local(
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
    )

    print(res)
