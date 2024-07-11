import os
import re
import json
import glob
import openai
from openai import OpenAI
import numpy as np
from huggingface_hub import InferenceClient
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import transformers
import torch
import time
from argparse import Namespace as AttrDict

import dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    wait_fixed,
)
import multiprocessing as mp

dotenv.load_dotenv(override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")
if os.getenv("OPENAI_API_TYPE") is not None:
    openai.api_type = os.getenv("OPENAI_API_TYPE")
if os.getenv("OPENAI_API_BASE") is not None:
    openai.api_base = os.getenv("OPENAI_API_BASE")
if os.getenv("OPENAI_API_VERSION") is not None:
    openai.api_version = os.getenv("OPENAI_API_VERSION")

LLAMA_TEMPLATE = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST] """


quantization_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

client = OpenAI(
    api_key=openai.api_key,
)


def get_hf_model_name(model_name):
    if model_name == "llama2-70b-chat":
        return "4bit/Llama-2-70b-chat-hf"
    elif model_name == "llama2-13b-chat":
        return "4bit/Llama-2-13b-chat-hf"
    elif model_name == "llama2-7b-chat":
        return "4bit/Llama-2-7b-chat-hf"
    else:
        return model_name


def load_hf_model(model_name, quantization_config):
    model_name = get_hf_model_name(model_name)

    if not torch.cuda.is_available():
        quantization_config = None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quantization_config, device_map="auto"
        )
    except:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, quantization_config=quantization_config, device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    model.config.eos_token_id = tokenizer.eos_token_id
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def flatten_conversation(convo):
    s = ""
    for c in convo:
        s += f"{c['role'].capitalize()}: {c['content']}\n\n"
    return s.strip()


LLAMA_TEMPLATE = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST]"""

MISTRAL_TEMPLATE = """<s> [INST] {user_message} [/INST]"""


def str_to_msgs(s):
    return [
        {"role": "system", "content": s.split("\n\n")[0]},
        {"role": "user", "content": "\n\n".join(s.split("\n\n")[1:])},
    ]


def encode_hfapi(input: list, model_name: str):
    assert 0 < len(input) <= 2

    if "llama" in model_name:
        if len(input) == 1:
            system_prompt = "You are a helpful assistant."
            assert input[0]["role"] == "user"
            user_message = input[0]["content"]
        elif len(input) == 2:
            assert input[0]["role"] == "system"
            system_prompt = input[0]["content"]
            assert input[1]["role"] == "user"
            user_message = input[1]["content"]
        else:
            raise ValueError("Unreachable code")

        return LLAMA_TEMPLATE.format(
            system_prompt=system_prompt, user_message=user_message
        )

    elif "mistral" in model_name or "mixtral" in model_name:
        if len(input) == 1:
            assert input[0]["role"] == "user"
            user_message = input[0]["content"]
        elif len(input) == 2:
            assert input[0]["role"] == "system"
            assert input[1]["role"] == "user"
            user_message = f"{input[0]['content']}\n\n{input[1]['content']}"
        else:
            raise ValueError("Unreachable code")

        return MISTRAL_TEMPLATE.format(user_message=user_message)

    else:
        raise ValueError("Invalid model name")


# @retry(
#   reraise=True,
#   stop=stop_after_attempt(20),
#   wait=wait_exponential(multiplier=1, min=1, max=20),
#   retry=(retry_if_exception_type(openai.Timeout)
#       | retry_if_exception_type(openai.APIError)
#       | retry_if_exception_type(openai.APIConnectionError)
#       | retry_if_exception_type(openai.RateLimitError)),


# )
def chatgpt_decode(
    input: list, model_name: str = "gpt-4", max_length: int = 128, temp: float = 0.3
):
    if openai.api_type == "azure":
        response = client.chat.completions.create(
            engine=model_name, messages=input, max_tokens=max_length, temperature=temp
        )
    else:
        response = client.chat.completions.create(
            model=model_name, messages=input, max_tokens=max_length, temperature=temp
        )

    # Assuming
    return response.choices[0].message.content
    # return [response.choices[j].message.content for j in range(len(response.choices))]


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_fixed(3))
def hfapi_decode(
    input: list,
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    max_length: int = 128,
    temp: float = 0.3,
):
    """Suggested models:
    - meta-llama/Llama-2-70b-chat-hf
    - mistralai/Mixtral-8x7B-Instruct-v0.1
    - mistralai/Mistral-7B-Instruct-v0.2
    """
    assert temp > 0

    input = encode_hfapi(input, model_name)
    model = InferenceClient(model=model_name, token=os.getenv("HUGGINGFACE_TOKEN"))
    output = model.post(
        json={
            "inputs": input,
            "parameters": {
                "max_new_tokens": max_length,
                "temperature": temp,
                "do_sample": True if temp > 0 else False,
            },
        }
    )
    text = json.loads(output)[0]["generated_text"].strip()
    if "[/INST]" in text:
        assert text.count("[/INST]") == 1
        text = text.split("[/INST]")[1].strip()
        return text
    return text
