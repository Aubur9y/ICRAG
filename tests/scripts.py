# 检查 Neo4j 中的节点和嵌入向量
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(
    "bolt://localhost:7687", auth=("neo4j", os.environ.get("NEO4J_PASSWORD"))
)
with driver.session() as session:
    # 获取一个节点的嵌入向量
    result = session.run(
        "MATCH (n:Entity) WHERE n.embedding IS NOT NULL RETURN n.name, n.embedding LIMIT 1"
    )
    record = result.single()
    if record:
        node_name = record["n.name"]
        node_embedding = record["n.embedding"]

        # 使用相同的嵌入向量作为查询
        query_embedding = node_embedding

        # 测试相似度计算
        result = session.run(
            """
        MATCH (n:Entity)
        WHERE n.embedding IS NOT NULL
        WITH n,
             reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) |
                    dot + n.embedding[i] * $query_embedding[i]) as dot_product,
             sqrt(reduce(sum = 0.0, i IN range(0, size(n.embedding)-1) |
                    sum + n.embedding[i] * n.embedding[i])) as node_norm,
             sqrt(reduce(sum = 0.0, i IN range(0, size($query_embedding)-1) |
                    sum + $query_embedding[i] * $query_embedding[i])) as query_norm
        WITH n, dot_product, node_norm, query_norm,
             CASE 
               WHEN node_norm > 0 AND query_norm > 0 
               THEN dot_product / (node_norm * query_norm)
               ELSE 0.0 
             END as similarity
        ORDER BY similarity DESC
        LIMIT 5
        RETURN n.name, similarity, node_norm, query_norm, dot_product
        """,
            query_embedding=query_embedding,
        )

        print(f"Testing with node: {node_name}")
        for record in result:
            print(
                f"Node: {record['n.name']}, Similarity: {record['similarity']:.6f}, "
                f"Node norm: {record['node_norm']:.6f}, Query norm: {record['query_norm']:.6f}, "
                f"Dot product: {record['dot_product']:.6f}"
            )

driver.close()
