import requests
import numpy as np
from config import (
    API_BASE_URL, API_KEY, API_CHAT_MODEL,
    EMBED_PROVIDER, API_EMBED_MODEL,
    OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL
)

# ---------- 通用 HTTP 请求头 ----------
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def api_chat(prompt, system="", temperature=0.3, format_json=False):
    """
    调用远程 API 的 Chat 接口 (OpenAI 兼容格式)
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": API_CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if format_json:
        payload["response_format"] = {"type": "json_object"}

    url = f"{API_BASE_URL}/chat/completions"
    response = requests.post(url, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def ollama_embed_local(text):
    """本地 Ollama Embedding 备选"""
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {"model": OLLAMA_EMBED_MODEL, "prompt": text}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["embedding"]


def api_embed(text):
    """调用远程 API 的 Embedding 接口"""
    url = f"{API_BASE_URL}/embeddings"
    payload = {
        "model": API_EMBED_MODEL,
        "input": text
    }
    response = requests.post(url, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def ollama_embed(text):
    """统一嵌入入口，根据配置选择来源"""
    if EMBED_PROVIDER == "api":
        return api_embed(text)
    else:
        return ollama_embed_local(text)


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def triple_to_text(triple):
    """三元组转自然语言片段"""
    if len(triple) == 3:
        return f"{triple[0]} {triple[1]} {triple[2]}"
    return str(triple)


# 为了兼容原有代码中的函数名
ollama_chat = api_chat