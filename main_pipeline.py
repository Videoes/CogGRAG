import json
import sys
import re
from pathlib import Path
import csv
import time

sys.path.append(str(Path.cwd()))

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    ENTITY_LABELS, RELATION_TYPES, SIMILARITY_THRESHOLD,
    API_CHAT_MODEL, API_BASE_URL
)
from kg_client import Neo4jClient
from utils import ollama_chat, ollama_embed, cosine_similarity, triple_to_text
from prompts import (
    DECOMPOSE_PROMPT_HEAD,
    EXTRACT_LOCAL_PROMPT,
    EXTRACT_GLOBAL_PROMPT,
    REASONING_PROMPT,
    SECOND_REASONING_PROMPT,
    RETHINK_PROMPT,
    fill_prompt
)

print("所有模块导入成功。")

# ---------- 步骤1：问题分解 ----------
def decompose_question(question, max_depth=5, current_depth=0):
    if current_depth >= max_depth:
        return [{"sub_question": question, "state": "End", "depth": current_depth, "children": []}]

    prompt = fill_prompt(DECOMPOSE_PROMPT_HEAD, question=question)
    response = ollama_chat(prompt, temperature=0.1, format_json=False)

    array_match = re.search(r'\[.*\]', response, re.DOTALL)
    if array_match:
        json_str = array_match.group(0)
        try:
            subqs = json.loads(json_str)
            if isinstance(subqs, list):
                valid_subqs = []
                for item in subqs:
                    if isinstance(item, dict) and "sub_question" in item and "state" in item:
                        valid_subqs.append(item)
                if valid_subqs:
                    subqs = valid_subqs
                else:
                    raise ValueError("数组中无有效子问题")
            else:
                raise ValueError("提取的内容不是列表")
        except Exception as e:
            print(f"解析数组失败：{e}")
            subqs = None
    else:
        obj_match = re.search(r'\{.*\}', response, re.DOTALL)
        if obj_match:
            try:
                obj = json.loads(obj_match.group(0))
                if isinstance(obj, dict) and "sub_question" in obj:
                    subqs = [obj]
                else:
                    raise ValueError("对象缺少 sub_question 字段")
            except Exception as e:
                print(f"解析对象失败：{e}")
                subqs = None
        else:
            subqs = None

    if subqs is None:
        return [{"sub_question": question, "state": "End", "depth": current_depth, "children": []}]

    result_nodes = []
    for item in subqs:
        sub_q = item.get("sub_question", "")
        state = item.get("state", "End")
        node = {
            "sub_question": sub_q,
            "state": state,
            "depth": current_depth,
            "children": []
        }
        if state == "Continue":
            node["children"] = decompose_question(sub_q, max_depth, current_depth + 1)
        else:
            node["children"] = []
        result_nodes.append(node)

    return result_nodes

def build_full_mind_map(original_question, max_depth=3):
    root = {
        "sub_question": original_question,
        "state": "Continue",
        "depth": 0,
        "children": []
    }
    root["children"] = decompose_question(original_question, max_depth, current_depth=1)
    return [root]

def collect_leaf_questions(nodes):
    leafs = []
    for node in nodes:
        if not node.get("children"):
            leafs.append(node["sub_question"])
        else:
            leafs.extend(collect_leaf_questions(node["children"]))
    return leafs

# ---------- 步骤2：实体提取 ----------
def extract_local_keys(question_list):
    prompt = fill_prompt(EXTRACT_LOCAL_PROMPT, mind_map=json.dumps(question_list, ensure_ascii=False))
    response = ollama_chat(prompt, temperature=0.1, format_json=False)
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        return {"entities": []}
    try:
        extracted = json.loads(json_match.group(0))
        result = {"entities": extracted.get("entities", [])}
        return result
    except:
        return {"entities": []}

# ---------- 步骤3：本地检索 ----------
MAX_HOPS = 2
TOP_K_NODES = 100

def retrieve_by_entity(entity_name, client, fuzzy=True, max_hops=MAX_HOPS, top_k_nodes=TOP_K_NODES):
    nodes = client.get_node_by_name(entity_name, fuzzy=fuzzy)
    if not nodes:
        return []
    all_triples = set()
    for node in nodes[:top_k_nodes]:
        node_id = node.get("element_id")
        if not node_id:
            continue
        if max_hops == 1:
            cypher = """
                MATCH (n)-[r]->(m)
                WHERE elementId(n) = $nid
                RETURN n.name AS start, type(r) AS relation, m.name AS end
                UNION
                MATCH (n)<-[r]-(m)
                WHERE elementId(n) = $nid
                RETURN m.name AS start, type(r) AS relation, n.name AS end
            """
        else:
            cypher = f"""
                MATCH path = (start)-[r*1..{max_hops}]-(end)
                WHERE elementId(start) = $nid
                UNWIND relationships(path) AS rel
                WITH startNode(rel) AS s, rel, endNode(rel) AS e
                RETURN s.name AS start, type(rel) AS relation, e.name AS end
            """
        results = client.query(cypher, {"nid": node_id})
        for rec in results:
            if rec["start"] and rec["relation"] and rec["end"]:
                all_triples.add((rec["start"], rec["relation"], rec["end"]))
    return list(all_triples)

def semantic_filter(candidate_triples, key_texts, threshold=SIMILARITY_THRESHOLD):
    if not candidate_triples:
        return []
    triple_texts = [triple_to_text(t) for t in candidate_triples]
    key_embeddings = [ollama_embed(text) for text in key_texts]
    filtered = []
    for triple, text in zip(candidate_triples, triple_texts):
        triple_emb = ollama_embed(text)
        max_sim = max(cosine_similarity(triple_emb, k_emb) for k_emb in key_embeddings)
        if max_sim >= threshold:
            filtered.append(triple)
    return filtered

def retrieve_local(entities, client):
    all_candidate_triples = set()
    for entity in entities:
        triples = retrieve_by_entity(entity, client, fuzzy=True)
        all_candidate_triples.update(triples)
    filtered = semantic_filter(list(all_candidate_triples), entities)
    return filtered

# ---------- 步骤4：全局检索 ----------
GLOBAL_TOP_K_NODES = 2

def extract_global_subgraph(question_list):
    prompt = fill_prompt(EXTRACT_GLOBAL_PROMPT, mind_map=json.dumps(question_list, ensure_ascii=False))
    response = ollama_chat(prompt, temperature=0.1, format_json=False)
    array_match = re.search(r'\[.*\]', response, re.DOTALL)
    if not array_match:
        return []
    try:
        triples = json.loads(array_match.group(0))
        if isinstance(triples, list):
            valid_triples = []
            for t in triples:
                if isinstance(t, list) and len(t) == 3:
                    valid_triples.append(t)
            return valid_triples
        else:
            return []
    except:
        return []

def match_global_subgraph(triple_patterns, client, max_results_per_pattern=5, top_k_nodes=GLOBAL_TOP_K_NODES):
    all_triples = set()
    for start, rel, end in triple_patterns:
        exact = client.match_triple(start, rel, end)
        if exact:
            all_triples.update(exact)
            continue
        nodes = client.get_node_by_name(start, fuzzy=True)
        if not nodes:
            continue
        found_any = False
        for node in nodes[:top_k_nodes]:
            node_id = node.get("element_id")
            if not node_id:
                continue
            cypher = """
                MATCH (n)-[r]->(m)
                WHERE elementId(n) = $nid AND type(r) = $rel
                RETURN n.name AS start, type(r) AS relation, m.name AS end
                LIMIT $limit
            """
            results = client.query(cypher, {"nid": node_id, "rel": rel, "limit": max_results_per_pattern})
            for rec in results:
                all_triples.add((rec["start"], rec["relation"], rec["end"]))
                found_any = True
        if not found_any:
            print(f"    未找到关系 '{rel}' 从 '{start}' 出发")
    return list(all_triples)

def retrieve_global(leaf_questions, client):
    patterns = extract_global_subgraph(leaf_questions)
    if not patterns:
        return []
    return match_global_subgraph(patterns, client)

# ---------- 步骤5：推理与综合 ----------
def format_knowledge(triples, max_len=None):
    lines = []
    for s, r, o in triples[:max_len]:
        lines.append(f"({s}, {r}, {o})")
    return "\n".join(lines)

def ensure_str(ans):
    if ans is None:
        return ""
    if isinstance(ans, list):
        return "、".join(str(x) for x in ans)
    return str(ans)

def first_reasoning(question, knowledge_triples, verified_answers):
    kg_text = format_knowledge(knowledge_triples, max_len=None)
    verified_text = json.dumps(verified_answers, ensure_ascii=False, indent=2)
    prompt = fill_prompt(REASONING_PROMPT, question=question,
                         knowledge_triples=kg_text, verified_answers=verified_text)
    response = ollama_chat(prompt, temperature=0.1, format_json=False)
    try:
        data = json.loads(response)
        ans = data.get("answer", "解析失败")
    except:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                ans = data.get("answer", "解析失败")
            except:
                ans = "解析失败"
        else:
            ans = "解析失败"
    return ensure_str(ans)

def second_reasoning(question, knowledge_triples, verified_answers):
    kg_text = format_knowledge(knowledge_triples, max_len=None)
    verified_text = json.dumps(verified_answers, ensure_ascii=False, indent=2)
    prompt = fill_prompt(SECOND_REASONING_PROMPT, question=question,
                         knowledge_triples=kg_text, verified_answers=verified_text)
    response = ollama_chat(prompt, temperature=0.2, format_json=False)
    try:
        data = json.loads(response)
        ans = data.get("answer", "解析失败")
    except:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                ans = data.get("answer", "解析失败")
            except:
                ans = "解析失败"
        else:
            ans = "解析失败"
    return ensure_str(ans)

def rethink(question, knowledge_triples, verified_answers):
    kg_text = format_knowledge(knowledge_triples, max_len=None)
    verified_text = json.dumps(verified_answers, ensure_ascii=False, indent=2)
    prompt = fill_prompt(RETHINK_PROMPT, question=question,
                         knowledge_triples=kg_text, verified_answers=verified_text)
    response = ollama_chat(prompt, temperature=0.1, format_json=False)
    try:
        data = json.loads(response)
        ans = data.get("answer", "解析失败")
    except:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                ans = data.get("answer", "解析失败")
            except:
                ans = "解析失败"
        else:
            ans = "解析失败"
    return ensure_str(ans)

def synthesize_final_answer(root_question, child_answers, knowledge_triples):
    kg_text = format_knowledge(knowledge_triples, max_len=None)
    answers_text = ""
    for q, ans in child_answers.items():
        answers_text += f"子问题：{q}\n答案：{ans}\n\n"
    prompt = f"""你是一个专业的知识整合与问答助手。你的任务是根据以下提供的子问题答案和相关知识三元组，对原始复杂问题生成一个完整、准确、自然的回答。

**回答要求：**
1. **综合所有子问题答案**：回答必须覆盖原始问题中的所有子问题要点，不能遗漏任何一个。
2. **结合知识图谱事实**：如果知识三元组中包含与问题相关的背景信息或细节，可以适当补充，使回答更加丰富、专业。
3. **严格基于给定信息**：禁止编造任何未在子问题答案或知识三元组中出现的内容。如果某些信息不足以支撑完整的叙述，请如实说明或留白，不要臆测。
4. **语言流畅自然**：用一段或几段连贯的文字表述，避免简单罗列或问答形式。使用中文输出。
5. **结构清晰**：可以按照问题逻辑分点陈述，但整体要融入段落中。

原始问题：{root_question}

已解答的子问题及对应答案：
{answers_text}

相关知识三元组（可作为补充背景）：
{kg_text}

知识图谱中的实体类别只有四种：动作要素及著作、文化要素及内涵、人物流派及机构、事件活动及项目。
- 动作要素及著作：指中医导引术中的具体动作、姿势、技法、著作（如“云手”、“揽雀尾”、“推手”、“中正”、“套路”、“舒展”、“吐纳术”、“震脚”、“太极操”、“《九要论》”、“《耍拳论》”）
- 文化要素及内涵：指中医导引术相关的文化概念、理论、术语（如“阴阳”“天人合一”“太极”）
- 人物流派及机构：指中医导引术相关的人物、流派、机构（如“杨露禅”“陈王廷”“太极拳”“陈氏太极拳”“五禽戏”“华佗五禽戏”）
- 事件活动及项目：指中医导引术相关的活动、事件、比赛、项目（如“世界太极拳健康大会”“全球健康促进大会”）

知识图谱中的关系类型只有四种：传承、展现、创编、参与。
- "展现" 可以表示模仿、包含、体现、记载等含义。
- "传承" 可以表示代表、继承、流传等含义。
- "创编" 表示创造、编著。
- "参与" 表示参加、涉及。

注意:传承关系中“主题-传承-客体”，如果主体和客体都是人物，则表示主体是客体的师傅、客体师从于客体。

请综合上述信息，给出最终答案："""
    print("      [综合] 正在调用 LLM 生成完整回答...")
    response = ollama_chat(prompt, temperature=0.3, format_json=False)
    print("      [综合] LLM 响应已接收。")
    return response.strip()

def postorder_traverse(nodes):
    for node in nodes:
        if node.get("children"):
            yield from postorder_traverse(node["children"])
        yield node

# ---------- 单问题处理（新增返回 contexts） ----------
def process_single_question(question, client):
    print(f"\n===== 处理问题：{question} =====\n")

    # Step 1: 分解
    print("Step 1: 构建思维导图...")
    mind_map = build_full_mind_map(question, max_depth=3)
    leaf_questions = collect_leaf_questions(mind_map)
    print(f"  根问题：{mind_map[0]['sub_question']}")
    print(f"  叶子问题：{leaf_questions}")

    # Step 2: 提取实体
    print("\nStep 2: 提取实体...")
    keys = extract_local_keys(leaf_questions)
    entities = keys.get("entities", [])
    print(f"  实体：{entities}")

    # Step 3: 本地检索
    print("\nStep 3: 本地知识检索...")
    local_triples = retrieve_local(entities, client)
    print(f"  本地检索获得 {len(local_triples)} 条三元组")

    # Step 4: 全局检索
    print("\nStep 4: 全局知识检索...")
    global_triples = retrieve_global(leaf_questions, client)
    print(f"  全局检索获得 {len(global_triples)} 条三元组")

    # 合并知识池
    combined_set = set()
    for t in local_triples:
        combined_set.add(tuple(t) if isinstance(t, list) else t)
    for t in global_triples:
        combined_set.add(tuple(t) if isinstance(t, list) else t)
    knowledge_triples = [list(t) for t in combined_set]
    print(f"  合并后知识池共 {len(knowledge_triples)} 条三元组")

    # 构建检索上下文（自然语言片段列表）
    contexts = [triple_to_text(t) for t in knowledge_triples]

    # Step 5: 推理与综合
    print("\nStep 5: 自底向上推理与综合...")
    verified_answers = {}
    for node in postorder_traverse(mind_map):
        q = node["sub_question"]
        if node["state"] == "Continue" or node.get("children"):
            continue
        print(f"\n  >>> 正在推理子问题：{q}")
        ans1 = first_reasoning(q, knowledge_triples, verified_answers)
        print(f"      第一次推理：{ans1[:80]}..." if len(ans1) > 80 else f"      第一次推理：{ans1}")
        ans2 = second_reasoning(q, knowledge_triples, verified_answers)
        print(f"      第二次推理：{ans2[:80]}..." if len(ans2) > 80 else f"      第二次推理：{ans2}")

        def is_insufficient(ans):
            return "信息不足" in ans or "无法回答" in ans

        if ans1.strip().lower() == ans2.strip().lower():
            if is_insufficient(ans1):
                print(f"      ⚠️ 两次答案一致但均为信息不足，触发重新思考...")
                final_ans = rethink(q, knowledge_triples, verified_answers)
                print(f"      🔄 重新思考答案：{final_ans[:80]}..." if len(final_ans) > 80 else f"      🔄 重新思考答案：{final_ans}")
            else:
                final_ans = ans1
                print(f"      ✅ 两次答案一致，采纳。")
        else:
            print(f"      ⚠️ 两次答案不一致，触发重新思考...")
            final_ans = rethink(q, knowledge_triples, verified_answers)
            print(f"      🔄 重新思考答案：{final_ans[:80]}..." if len(final_ans) > 80 else f"      🔄 重新思考答案：{final_ans}")

        verified_answers[q] = final_ans

    root_node = mind_map[0]
    root_question = root_node["sub_question"]
    child_answers = {q: ans for q, ans in verified_answers.items() if q != root_question}
    print(f"\n  >>> 正在综合生成最终答案（整合所有子问题答案）...")
    final_answer = synthesize_final_answer(root_question, child_answers, knowledge_triples)
    verified_answers[root_question] = final_answer
    print(f"\n最终答案：\n{final_answer}")

    return final_answer, verified_answers, knowledge_triples, contexts

# ---------- 批量处理（写入 CSV 时包含检索上下文） ----------
def process_batch(input_csv_path, output_csv_path, client):
    questions = []
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row and row[1].strip():
                questions.append((row[0], row[1].strip()))

    print(f"共读取 {len(questions)} 个问题。")

    processed_ids = set()
    if Path(output_csv_path).exists():
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    processed_ids.add(row[0])

    for idx, q in questions:
        if idx in processed_ids:
            print(f"问题 {idx} 已处理，跳过。")
            continue
        print(f"\n{'='*50}\n处理序号 {idx}：{q}\n{'='*50}")
        try:
            final_answer, _, _, contexts = process_single_question(q, client)
        except Exception as e:
            print(f"处理失败：{e}")
            final_answer = f"处理出错：{e}"
            contexts = []

        context_str = json.dumps(contexts, ensure_ascii=False) if contexts else "[]"

        file_exists = Path(output_csv_path).exists()
        with open(output_csv_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["序号", "问题", "综合回答", "检索上下文"])
            writer.writerow([idx, q, final_answer, context_str])
        print(f"结果已写入 {output_csv_path}")

# ---------- 主程序 ----------
if __name__ == "__main__":
    client = Neo4jClient()
    try:
        result = client.query("MATCH (n) RETURN count(n) AS total")
        total_nodes = result[0]['total'] if result else 0
        print(f"Neo4j 连接成功！节点总数：{total_nodes}")
    except Exception as e:
        print(f"Neo4j 连接失败：{e}")
        client = None
        exit(1)

    print("\n请选择运行模式：")
    print("1. 单个问题处理")
    print("2. 批量处理（从 data/待解决问题.csv 读取，写入 data/已解决问题.csv）")
    mode = input("输入 1 或 2：").strip()

    if mode == "1":
        question = input("请输入问题：").strip()
        if question:
            process_single_question(question, client)
        else:
            print("问题不能为空。")
    elif mode == "2":
        input_file = "data/待解决问题.csv"
        output_file = "data/已解决问题.csv"
        if not Path(input_file).exists():
            print(f"错误：输入文件 {input_file} 不存在。")
        else:
            process_batch(input_file, output_file, client)
    else:
        print("无效选择。")

    client.close()
    print("处理完成。")