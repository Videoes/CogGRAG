import requests
import numpy as np
from config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL

def ollama_chat(prompt, system="", temperature=0.3, format_json=False):
    """
    调用 Ollama Chat API
    :param prompt: 用户输入文本
    :param system: 系统提示（可选）
    :param temperature: 采样温度
    :param format_json: 是否强制返回 JSON 格式
    :return: 模型生成的文本内容
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature ,
        }
    }
    if format_json:
        payload["format"] = "json"

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["message"]["content"]

def ollama_embed(text):
    """
    调用 Ollama Embeddings API
    :param text: 待编码文本
    :return: 向量列表 (list of floats)
    """
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "prompt": text
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["embedding"]

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def triple_to_text(triple):
    """
    将三元组（如 ('a', 'r', 'b')）转换为自然语言描述，便于嵌入计算
    """
    if len(triple) == 3:
        return f"{triple[0]} {triple[1]} {triple[2]}"
    else:
        return str(triple)