# ==================== Neo4j 连接配置 ====================
NEO4J_URI = ""
NEO4J_USER = ""
NEO4J_PASSWORD = ""

# ==================== 远程 API 配置 (moyu.info) ====================
API_BASE_URL = ""          # OpenAI 兼容端点
API_KEY = ""
API_CHAT_MODEL = ""

# ==================== Embedding 配置 ====================
# 如果远程 API 支持 embeddings，可填写相应模型名；否则回退本地 Ollama
EMBED_PROVIDER = "local"               # "api" 或 "local"
API_EMBED_MODEL = ""   # 若 API 支持
OLLAMA_EMBED_MODEL = ""
OLLAMA_BASE_URL = ""

# ==================== 检索阈值 ====================
SIMILARITY_THRESHOLD = 0.3

# ==================== 图谱 Schema 常量 ====================
ENTITY_LABELS = [
    "动作要素及著作",
    "文化要素及内涵",
    "人物流派及机构",
    "事件活动及项目"
]

RELATION_TYPES = [
    "传承",
    "展现",
    "创编",
    "参与"
]

ALLOWED_PATTERNS = {
    ("人物流派及机构", "传承", "文化要素及内涵"): True,
    ("人物流派及机构", "传承", "人物流派及机构"): True,
    ("动作要素及著作", "展现", "文化要素及内涵"): True,
    ("事件活动及项目", "展现", "文化要素及内涵"): True,
    ("人物流派及机构", "展现", "动作要素及著作"): True,
    ("人物流派及机构", "创编", "动作要素及著作"): True,
    ("人物流派及机构", "参与", "事件活动及项目"): True,
}