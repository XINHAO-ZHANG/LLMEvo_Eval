"""
llm_ops.api
===========

Single entry-point `call_llm()` used everywhere in GA loop.
Supports:
  • OpenAI  (gpt-3.5-turbo, gpt-4o,…)
  • Azure-OpenAI  (model endswith ":azure")
  • Ollama / local http://127.0.0.1:11434  (model name自由)

Install deps:
  pip install openai requests
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, Any, List, Union

import requests
from openai import OpenAI, AzureOpenAI
import openai

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Client Initialization (using environment variables)
# --------------------------------------------------------------------------- #

load_dotenv()

# Cerebras Client
cerebras_client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))

# openrouter client
openrouter_client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url=os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1"))

# OpenAI Client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("OPENAI_API_URL", "https://api.openai.com"))

# Azure OpenAI Client
azure_client = None
if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
    azure_client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01") # Use a default or env var
    )

# --------------------------------------------------------------------------- #
# Provider detection helper
# --------------------------------------------------------------------------- #
def _provider_from(model_name: str) -> str:
    if model_name.endswith(":azure"): return "azure"
    # Basic check for common local model names, adjust as needed
    if model_name.startswith(("llama")):
        return "cerebras"
    if model_name.startswith(("deepseek", "qwen", "meta-llama", "x-ai", "mistralai", "minimax", "google")):
        return "openrouter"
    return "openai"

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
    max_retries: int = 5,  # 添加重试次数参数
) -> Dict[str, Any]:
    """
    Generic LLM caller. Returns a dict that **必须** 至少有键:
        - "genome" 或 "text"
        - "usage"  (dict 包含 total_tokens)
    下游 GA 会用到 usage（计费）以及 genome/text。

    `prompt_or_msgs`:
        • str  → 传统 prompt
        • list[{"role": "...", "content": "..."}] → Chat 完整消息
        
    重试机制：
        • 最多重试 max_retries 次
        • 在连接错误、超时或API错误时重试
    """
    prov = _provider_from(model)
    model_name = model.split(":azure")[0] # Get base model name
    
    # 实现重试逻辑
    retry_count = 0
    last_error = None
    
    while retry_count <= max_retries:
        try:
            if prov == "openai":
                if not openai_client.api_key:
                     raise EnvironmentError("OPENAI_API_KEY missing or OpenAI client not initialized!")
                return _call_openai(openai_client, prompt_or_msgs, model_name, temperature, max_tokens, seed)
            elif prov == "cerebras":
                if not cerebras_client:
                    raise EnvironmentError("Cerebras OpenAI environment variables missing or Cerebras client not initialized!")
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
            print(f"LLM call failed ( attempt {retry_count}/{max_retries}): {str(e)}")
            
            if retry_count > max_retries:
                break
                
            # 简单的退避策略
            import time
            time.sleep(2 * retry_count)  # 递增等待时间
            continue
    
    # 如果所有重试都失败，返回一个基本结构以避免下游处理错误
    print(f"警告: 所有LLM调用尝试都失败: {str(last_error)}")
    return {
        "text": f"LLM调用失败: {str(last_error)}",
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
#   cerebras
# ────────────────────────────────────────────────────────────────────────────
def _call_cerebras(client: Cerebras, prompt, model, temperature, max_tokens, seed):
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

    # try:
    #     data = json.loads(content)
    # except Exception:
    #     # 尝试截取 ```json ... ``` 
    #     import re, textwrap
    #     m = re.search(
    # r'(?:```json(.*?)```|<think>\s*</think>\s*({.*?}))',
    # content,
    # re.S)
    #     if m:
    #         try:
    #             data = json.loads(m.group(1) or m.group(2))
    #         except Exception:
    #             print(f"JSON parsing failed: {m.group(1) or m.group(2)}")
    #             data = {"text": content}
    #     else:
    #         print(f"JSON parsing failed: {content}")
    #         data = {"text": content}
    data = {"text": content}
    data["usage"] = usage or {"total_tokens": 0} # Provide default usage if None
    return data


# ────────────────────────────────────────────────────────────────────────────
#  Azure-OpenAI
# ────────────────────────────────────────────────────────────────────────────
def _call_azure(client: AzureOpenAI, prompt, deployment_name, temperature, max_tokens, seed):
    kwargs = dict(
        model=deployment_name, # Use model instead of engine
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

    # Handle list of messages - simple concatenation for now
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

        # Use more accurate token counts if available
        prompt_tokens = resp_json.get("prompt_eval_count", 0)
        completion_tokens = resp_json.get("eval_count", 0)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }

    except requests.exceptions.RequestException as e:
        print(f"Ollama request failed: {e}")
        # Fallback or re-raise
        content = ""
        usage = {"total_tokens": 0}
    except json.JSONDecodeError as e:
        print(f"Failed to decode Ollama JSON response: {e}")
        content = response.text # Store raw text if JSON fails
        usage = {"total_tokens": 0}


    try:
        data = json.loads(content)
    except Exception:
        data = {"text": content}
    data["usage"] = usage
    return data