"""
llm_ops.api
===========

Single entry-point `call_llm()` used everywhere in the GA loop.
Supports:
  • OpenAI  (gpt-3.5-turbo, gpt-4o, …)
  • Azure-OpenAI  (model endswith ":azure")
  • Ollama / local http://127.0.0.1:11434  (any model name)
  • OpenRouter, HuggingFace, Cerebras

Install deps:
  pip install openai requests
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Union

import requests
from openai import OpenAI, AzureOpenAI
import openai

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:
    Cerebras = None  # type: ignore[misc, assignment]

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Client Initialization (using environment variables)
# --------------------------------------------------------------------------- #

load_dotenv()

# HuggingFace Client
huggingface_client = None
if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    huggingface_client = OpenAI(
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        base_url=os.getenv("HUGGINGFACEHUB_API_URL", "https://router.huggingface.co/v1")
    )

# Cerebras Client (optional)
cerebras_client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY")) if Cerebras and os.getenv("CEREBRAS_API_KEY") else None

# OpenRouter client (optional; required for mistralai, deepseek, meta-llama, etc.)
openrouter_client = None
if os.getenv("OPENROUTER_API_KEY"):
    openrouter_client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
    )

# OpenAI Client (optional)
openai_client = None
if os.getenv("OPENAI_API_KEY"):
    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_URL", "https://api.openai.com")
    )

# Azure OpenAI Client
azure_client = None
if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
    azure_client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    )

# --------------------------------------------------------------------------- #
# Provider detection helper
# --------------------------------------------------------------------------- #
def _provider_from(model_name: str) -> str:
    if model_name.endswith(":azure"): return "azure"
    if model_name.startswith(("Qwen", "IDinsight")) or model_name.endswith(":cerebras"):
        return "huggingface"
    if model_name.startswith(("deepseek", "meta-llama", "x-ai", "mistralai", "minimax", "google")):
        return "openrouter"
    return "openai"


def get_llm_provider(model_name: str) -> str:
    """Return the provider id for a given model name (e.g. 'openrouter', 'openai')."""
    return _provider_from(model_name)


# --------------------------------------------------------------------------- #
# Main API
# --------------------------------------------------------------------------- #
def call_llm(
    prompt_or_msgs: Union[str, List[Dict[str, str]]],
    *,
    model: str,
    max_tokens: int = 120,
    temperature: float = 0.7,
    seed: int | None = None,
    max_retries: int = 5,
) -> Dict[str, Any]:
    """
    Generic LLM caller. Returns a dict with at least:
        - "text"   : the model's response string
        - "usage"  : dict containing total_tokens

    `prompt_or_msgs`:
        • str  → plain prompt
        • list[{"role": "...", "content": "..."}] → full chat messages

    Retry policy: up to max_retries attempts with linear back-off on
    connection errors, timeouts, and API errors.
    """
    prov = _provider_from(model)
    model_name = model.split(":azure")[0]

    retry_count = 0
    last_error = None

    while retry_count <= max_retries:
        try:
            if prov == "openai":
                if not openai_client:
                    raise EnvironmentError("OPENAI_API_KEY missing or OpenAI client not initialized!")
                return _call_openai(openai_client, prompt_or_msgs, model_name, temperature, max_tokens, seed)
            elif prov == "huggingface":
                if not huggingface_client:
                    raise EnvironmentError("HuggingFace environment variables missing or HuggingFace client not initialized!")
                return _call_huggingface(huggingface_client, prompt_or_msgs, model_name, temperature, max_tokens, seed)
            elif prov == "cerebras":
                if not cerebras_client:
                    raise EnvironmentError("Cerebras environment variables missing or Cerebras client not initialized!")
                return _call_cerebras(cerebras_client, prompt_or_msgs, model_name, temperature, max_tokens, seed)
            elif prov == "openrouter":
                if not openrouter_client:
                    raise EnvironmentError("OpenRouter environment variables missing or OpenRouter client not initialized!")
                return _call_openrouter(openrouter_client, prompt_or_msgs, model_name, temperature, max_tokens, seed)
            elif prov == "azure":
                if not azure_client:
                    raise EnvironmentError("Azure OpenAI environment variables missing or Azure client not initialized!")
                return _call_azure(azure_client, prompt_or_msgs, model_name, temperature, max_tokens, seed)
            elif prov == "ollama":
                return _call_ollama(prompt_or_msgs, model_name, temperature, max_tokens, seed)
            else:
                raise ValueError(f"Unknown provider for model '{model}'")
        except (requests.exceptions.RequestException, openai.APIError, openai.RateLimitError,
                openai.APIConnectionError, json.JSONDecodeError, ValueError) as e:
            last_error = e
            retry_count += 1
            print(f"LLM call failed (attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count > max_retries:
                break
            import time
            time.sleep(2 * retry_count)  # linear back-off
            continue

    print(f"Warning: all LLM call attempts failed: {str(last_error)}")
    return {
        "text": f"LLM call failed: {str(last_error)}",
        "error": str(last_error),
        "usage": {"total_tokens": 0}
    }

# ────────────────────────────────────────────────────────────────────────────
#  OpenRouter
# ────────────────────────────────────────────────────────────────────────────
def _call_openrouter(client: OpenAI, prompt, model, temperature, max_tokens, seed):
    kwargs = dict(model=model, temperature=temperature, max_tokens=max_tokens)
    if seed is not None:
        kwargs["seed"] = seed

    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]

    resp = client.chat.completions.create(messages=messages, **kwargs)
    content = resp.choices[0].message.content
    usage = resp.usage.to_dict() if resp.usage else None

    data = {"text": content}
    data["usage"] = usage or {"total_tokens": 0}
    return data

# ────────────────────────────────────────────────────────────────────────────
#  Cerebras
# ────────────────────────────────────────────────────────────────────────────
def _call_cerebras(client: Any, prompt, model, temperature, max_tokens, seed):
    kwargs = dict(model=model, temperature=temperature, max_tokens=max_tokens)
    if seed is not None:
        kwargs["seed"] = seed

    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]

    resp = client.chat.completions.create(messages=messages, **kwargs)
    content = resp.choices[0].message.content
    usage = resp.usage.to_dict() if resp.usage else None

    data = {"text": content}
    data["usage"] = usage or {"total_tokens": 0}
    return data

# ────────────────────────────────────────────────────────────────────────────
#  HuggingFace
# ────────────────────────────────────────────────────────────────────────────
def _call_huggingface(client, prompt, model, temperature, max_tokens, seed):
    kwargs = dict(model=model, temperature=temperature, max_tokens=max_tokens)
    if seed is not None:
        kwargs["seed"] = seed

    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]

    resp = client.chat.completions.create(messages=messages, **kwargs)
    content = resp.choices[0].message.content
    usage = resp.usage.to_dict() if resp.usage else None

    data = {"text": content}
    data["usage"] = usage or {"total_tokens": 0}
    return data

# ────────────────────────────────────────────────────────────────────────────
#  OpenAI
# ────────────────────────────────────────────────────────────────────────────
def _call_openai(client: OpenAI, prompt, model, temperature, max_tokens, seed):
    kwargs = dict(model=model, temperature=temperature, max_tokens=max_tokens)
    if seed is not None:
        kwargs["seed"] = seed

    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]

    resp = client.chat.completions.create(messages=messages, **kwargs)
    content = resp.choices[0].message.content
    usage = resp.usage.to_dict() if resp.usage else None

    data = {"text": content}
    data["usage"] = usage or {"total_tokens": 0}
    return data


# ────────────────────────────────────────────────────────────────────────────
#  Azure-OpenAI
# ────────────────────────────────────────────────────────────────────────────
def _call_azure(client: AzureOpenAI, prompt, deployment_name, temperature, max_tokens, seed):
    kwargs = dict(
        model=deployment_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if seed is not None:
        kwargs["seed"] = seed

    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    resp = client.chat.completions.create(messages=messages, **kwargs)
    content = resp.choices[0].message.content
    usage = resp.usage.to_dict() if resp.usage else None

    try:
        data = json.loads(content)
    except Exception:
        data = {"text": content}
    data["usage"] = usage or {"total_tokens": 0}
    return data


# ────────────────────────────────────────────────────────────────────────────
#  Ollama / local models
# ────────────────────────────────────────────────────────────────────────────
def _call_ollama(prompt, model, temperature, max_tokens, seed):
    url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")

    if isinstance(prompt, list):
        prompt_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in prompt])
    else:
        prompt_str = prompt

    payload = {
        "model": model,
        "prompt": prompt_str,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "seed": seed or random.randint(0, 2**31 - 1),
        },
        "stream": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        resp_json = response.json()
        content = resp_json.get("response", "")

        prompt_tokens = resp_json.get("prompt_eval_count", 0)
        completion_tokens = resp_json.get("eval_count", 0)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }

    except requests.exceptions.RequestException as e:
        print(f"Ollama request failed: {e}")
        content = ""
        usage = {"total_tokens": 0}
    except json.JSONDecodeError as e:
        print(f"Failed to decode Ollama JSON response: {e}")
        content = response.text
        usage = {"total_tokens": 0}

    try:
        data = json.loads(content)
    except Exception:
        data = {"text": content}
    data["usage"] = usage
    return data
