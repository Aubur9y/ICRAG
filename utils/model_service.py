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


if __name__ == "__main__":
    # res = chat_with_model(
    #     "sk-a49b9000d26f46efbb5731b3cbe91636",
    #     "llama4:latest",
    #     messages=[{"role": "user", "content": "Why is the sky blue?"}],
    # )
    #
    # print(res)

    res = chat_with_model(
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
    )

    print(res)
