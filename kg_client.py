from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ENTITY_LABELS, RELATION_TYPES

class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def query(self, cypher, params=None):
        """
        执行 Cypher 查询并返回结果列表
        """
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

    def get_node_by_name(self, name, fuzzy=True):
        label_clause = " OR ".join([f"n:`{label}`" for label in ENTITY_LABELS])
        if fuzzy:
            cypher = f"""
                MATCH (n)
                WHERE ({label_clause})
                AND n.name CONTAINS $name
                RETURN n, elementId(n) AS nid
                LIMIT 10
            """
        else:
            cypher = f"""
                MATCH (n)
                WHERE ({label_clause})
                AND n.name = $name
                RETURN n, elementId(n) AS nid
                LIMIT 1
            """
        results = self.query(cypher, {"name": name})
        nodes = []
        for record in results:
            node_data = record.get("n", {})
            node_data["element_id"] = record.get("nid")
            nodes.append(node_data)
        print(f"      [Neo4j] 查找 '{name}' (fuzzy={fuzzy}) 返回 {len(nodes)} 个节点")
        return nodes
    
    def get_triples_for_entity(self, entity_name):
        """
        获取实体的一跳邻居三元组（格式为 (start, relation, end)）
        """
        # 先通过模糊匹配找到节点
        nodes = self.get_node_by_name(entity_name, fuzzy=True)
        if not nodes:
            return []

        # 取第一个匹配的节点 ID
        node_id = nodes[0].get("id") or nodes[0].element_id  # Neo4j 5+ 使用 element_id
        cypher = """
            MATCH (n)-[r]->(m)
            WHERE elementId(n) = $nid
            RETURN n.name AS start, type(r) AS relation, m.name AS end
            UNION
            MATCH (n)<-[r]-(m)
            WHERE elementId(n) = $nid
            RETURN m.name AS start, type(r) AS relation, n.name AS end
        """
        results = self.query(cypher, {"nid": node_id})
        triples = [(rec["start"], rec["relation"], rec["end"]) for rec in results]
        return triples

    def match_triple(self, start_name, rel_type, end_name):
        """
        精确匹配一个三元组
        """
        cypher = """
            MATCH (a)-[r]->(b)
            WHERE a.name = $start AND type(r) = $rel AND b.name = $end
            RETURN a.name AS start, type(r) AS relation, b.name AS end
        """
        results = self.query(cypher, {"start": start_name, "rel": rel_type, "end": end_name})
        return [(rec["start"], rec["relation"], rec["end"]) for rec in results]

    def match_subgraph(self, triple_patterns):
        """
        匹配一个子图模式列表，每个模式为 (start, rel, end) 形式
        返回所有匹配的路径三元组列表
        """
        # 简单实现：逐个模式匹配并合并结果（可优化为复杂子图查询）
        all_triples = []
        for start, rel, end in triple_patterns:
            matched = self.match_triple(start, rel, end)
            all_triples.extend(matched)
        return all_triples