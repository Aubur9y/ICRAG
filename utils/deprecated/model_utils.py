# ========================
# This file is not used in the final product, only in the early stages for exploration.
# ========================


from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import os
import dotenv

dotenv.load_dotenv("../config.env")


def initialise_llm(model, temperature=0):
    """
    Initializes the LLM with the specified model and temperature.

    Args:
        model (str): The model to use for the LLM.
        temperature (float): The temperature setting for the LLM.

    Returns:
        ChatOpenAI: An instance of the ChatOpenAI class configured with the specified model and temperature.
    """
    if not model:
        raise ValueError("Model must be specified")
    if temperature < 0 or temperature > 1:
        raise ValueError("Temperature must be between 0 and 1")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable must be set")

    api_key_str = os.getenv("OPENAI_API_KEY")
    api_key_secret = SecretStr(api_key_str) if api_key_str else None

    return ChatOpenAI(api_key=api_key_secret, model=model, temperature=temperature)
