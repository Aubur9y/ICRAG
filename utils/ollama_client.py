import ollama
import requests
import time


class OllamaClient:
    def __init__(self, host="http://localhost:11434", max_retries=3, retry_delay=1):
        self.host = host
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def embed(self, model, input_text):
        for attempt in range(self.max_retries):
            try:
                return ollama.embed(model=model, input=input_text)
            except (requests.RequestException, ConnectionError) as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise e

    def chat(self, model, message):
        for attempt in range(self.max_retries):
            try:
                return ollama.chat(model=model, messages=message)
            except (requests.RequestException, ConnectionError) as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return e
