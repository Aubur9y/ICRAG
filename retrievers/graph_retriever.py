import json
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from utils.ollama_client import OllamaClient

import networkx as nx
from openai import OpenAI

from utils.file_embedding import (
    process_code_file,
    process_text_file,
    process_pdf_file,
    process_ipynb_file,
)
from neo4j import GraphDatabase

from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

database_config = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": os.environ.get("NEO4J_PASSWORD"),
}


class Neo4jConnection:
    def __init__(
        self,
        uri=database_config["uri"],
        user=database_config["user"],
        password=database_config["password"],
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self.driver:
            self.driver.close()

    def query(self, query, parameters=None, db=None):
        assert self.driver is not None, "Driver not initialised"
        session = None
        response = None

        try:
            session = (
                self.driver.session(database=db)
                if db is not None
                else self.driver.session()
            )
            response = list(session.run(query, parameters))
        except Exception as e:
            logging.error(f"Query failed: {e}")
        finally:
            if session is not None:
                session.close()
        return response

    def create_node(self, node_id, properties):
        query = """
        MERGE (n:Entity {id: $node_id})
        SET n += $properties
        RETURN n
        """
        return self.query(query, {"node_id": node_id, "properties": properties})

    def create_relationship(self, source_id, target_id, rel_type, properties=None):
        query = """
        MATCH (a:Entity {id: $source_id})
        MATCH (b:Entity {id: $target_id})
        MERGE (a)-[r:RELATES {type: $rel_type}]->(b)
        SET r += $properties
        RETURN r
        """
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "rel_type": rel_type,
            "properties": properties or {},
        }
        return self.query(query, params)

    def search_similar_nodes(self, query_embedding, k=3):
        query = """
        MATCH (n:Entity)
        WHERE n.embedding IS NOT NULL
        WITH n,
             reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) |
                    dot + n.embedding[i] * $query_embedding[i]) as similarity
        ORDER BY similarity DESC
        LIMIT toInteger($k)
        RETURN n, similarity
        """
        return self.query(query, {"query_embedding": query_embedding, "k": k})


class KnowledgeGraphBuilder:
    def __init__(self):
        self.client = OpenAI()
        self.graph = nx.DiGraph()

    def extract_entities_and_relations(self, text):
        # extract entities and relations from text
        text_truncated = text[:1500] if len(text) > 1500 else text

        prompt = f"""Extract entities and relations from the following text. Return ONLY a valid JSON object without any additional text or formatting:

        {{"entities": [{{"name": "entity_name", "type": "entity_type", "description": "brief_description"}}], "relations": [{{"source": "source_entity", "target": "target_entity", "relation": "relation_type"}}]}}

        Text: {text_truncated}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            response_text = response.choices[0].message.content

            if not response_text:
                raise ValueError("No content returned from OpenAI API")
            response_text = response_text.strip()

            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()

            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")

        except Exception as e:
            print(f"Error extracting entities: {e}")
            return {"entities": [], "relations": []}

    def extract_code_entities(self, code_content, language):
        # extract entities and relations from code
        code_truncated = (
            code_content[:1500] if len(code_content) > 1500 else code_content
        )

        prompt = f"""Extract programming entities and dependencies from the following {language} code. Return ONLY a valid JSON object without any additional text or formatting:

        {{"entities": [{{"name": "entity_name", "type": "function", "description": "functionality_description"}}], "relations": [{{"source": "source_entity", "target": "target_entity", "relation": "calls"}}]}}

        Code: {code_truncated}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            response_text = response.choices[0].message.content

            if not response_text:
                raise ValueError("No content returned from OpenAI API")

            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()

            # Try to find JSON in the response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")

        except Exception as e:
            print(f"Error extracting code entities: {e}")
            return {"entities": [], "relations": []}

    def build_graph_from_chunks(self, chunks):
        # build knowledge graph from document chunks
        def process_chunk(chunk):
            content = chunk["content"]
            chunk_type = chunk["type"]
            try:
                if chunk_type == "code":
                    language = chunk.get("language", "unknown")
                    result = self.extract_code_entities(content, language)
                else:
                    result = self.extract_entities_and_relations(content)

                if result is None:
                    result = {"entities": [], "relations": []}
                return (chunk, result)
            except Exception as e:
                print(f"Chunk {chunk.get('id', 'unknown')} failed: {e}")
                return (chunk, {"entities": [], "relations": []})

        # multi-threading
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_chunk = {
                executor.submit(process_chunk, chunk): chunk for chunk in chunks
            }
            # return
            for future in as_completed(future_to_chunk):
                try:
                    chunk, entities_relations = future.result()
                except Exception as e:
                    chunk = future_to_chunk[future]
                    print(f"Chuhnk {chunk.get('id', 'unknown')} failed: {e}")
                    entities_relations = {"entities": [], "relations": []}
                results.append((chunk, entities_relations))

        for chunk, entities_relations in results:
            # content = chunk['content']
            # chunk_type = chunk['type']

            # if chunk_type == 'code':
            #     language = chunk.get('language', 'unknown')
            #     entities_relations = self.extract_code_entities(content, language)
            # else:
            #     entities_relations = self.extract_entities_and_relations(content)

            # add entity nodes
            for entity in entities_relations["entities"]:
                node_id = f"{entity['name']}_{entity['type']}"
                self.graph.add_node(
                    node_id,
                    name=entity["name"],
                    type=entity["type"],
                    description=entity.get("description", ""),
                    source_chunk=chunk["id"],
                    content_type=chunk["type"],
                )

            # add relation edges
            for relation in entities_relations["relations"]:
                source_id = self.find_or_create_node(relation["source"])
                target_id = self.find_or_create_node(relation["target"])

                self.graph.add_edge(
                    source_id,
                    target_id,
                    relation=relation["relation"],
                    source_chunk=chunk["id"],
                )
        return self.graph

    # this is a helper function to add relation edges, if no target node then create one
    def find_or_create_node(self, entity_name, entity_type=None):
        # try to find existing node
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if node_data.get("name") == entity_name and (
                entity_type is None or node_data.get("type") == entity_type
            ):
                return node_id

        # create new node if not found
        entity_type = self.infer_entity_type(entity_name)
        node_id = f"{entity_name}_{entity_type}"
        self.graph.add_node(
            node_id,
            name=entity_name,
            type=entity_type,
            description="",
            source_chunk="unknown",
            content_type="unknown",
        )
        return node_id

    # also a helper function to infer the type of the entry based on the entity name
    def infer_entity_type(self, entity_name: str) -> str:
        # infer entity type using simple heuristics
        if any(
            keyword in entity_name.lower() for keyword in ["function", "def", "method"]
        ):
            return "function"
        elif any(keyword in entity_name.lower() for keyword in ["class", "object"]):
            return "class"
        elif any(
            keyword in entity_name.lower()
            for keyword in ["import", "library", "module"]
        ):
            return "import"
        else:
            return "concept"


class NodeEmbedder:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.embedding_model = "mxbai-embed-large"

    def generate_node_embedding(self, node_data):
        try:
            # generate embedding for graph node
            node_text = self.create_node_text(node_data)
            responses = self.ollama_client.embed(
                model=self.embedding_model, input_text=node_text
            )
            if (
                responses
                and "embeddings" in responses
                and len(responses["embeddings"]) > 0
            ):
                return responses["embeddings"][0]
            else:
                raise ValueError("Invalid embedding response")
        except Exception as e:
            logging.error(f"Embedding generation failed: {e}")
            return None

    # helper function to create a piece of text including name, type and description for each node
    def create_node_text(self, node_data):
        name = node_data.get("name", "")
        entity_type = node_data.get("type", "")
        description = node_data.get("description", "")

        return f"Entity: {name}, Type: {entity_type}, Description: {description}"

    # generate embedding for every node in a graph
    def embed_all_nodes(self, graph):
        node_embeddings = {}

        def embed_one(node_id, node_data):
            return node_id, self.generate_node_embedding(node_data)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(embed_one, node_id, node_data): node_id
                for node_id, node_data in graph.nodes(data=True)
            }
            for future in as_completed(futures):
                node_id, embedding = future.result()
                node_embeddings[node_id] = embedding

        # for node_id, node_data in graph.nodes(data=True):
        #     embedding = self.generate_node_embedding(node_data)
        #     node_embeddings[node_id] = embedding
        return node_embeddings


class GraphRAGRetriever:
    def __init__(self, neo4j_conn):
        self.neo4j_conn = neo4j_conn
        self.embedder = NodeEmbedder()

    def store_graph_to_neo4j(self, graph, node_embeddings):
        for node_id, node_data in graph.nodes(data=True):
            properties = dict(node_data)
            if node_id in node_embeddings:
                properties["embedding"] = node_embeddings[node_id]
            self.neo4j_conn.create_node(node_id, properties)

        for source, target, edge_data in graph.edges(data=True):
            rel_type = edge_data.get("relation", "RELATES_TO")
            self.neo4j_conn.create_relationship(source, target, rel_type, edge_data)

    def retrieve_relevant_subgraph(self, query, k=3, hops=2):
        response = self.embedder.ollama_client.embed(
            model=self.embedder.embedding_model, input_text=query
        )
        if not response or "embeddings" not in response or not response["embeddings"]:
            raise ValueError("No embedding returned for query")
        query_embedding = response["embeddings"][0]

        similar_nodes = self.neo4j_conn.search_similar_nodes(query_embedding, k)

        if not similar_nodes:
            return {"context": "No relevant nodes found", "similarity_scores": []}

        relevant_node_ids = [record["n"]["id"] for record in similar_nodes]
        similarity_scores = [record["similarity"] for record in similar_nodes]

        expanded_nodes = self.expand_subgraph(relevant_node_ids, hops)

        context = self.extract_context_from_neo4j(expanded_nodes, similarity_scores)

        return {
            "relevant_nodes": relevant_node_ids,
            "context": context,
            "similarity_scores": similarity_scores,
        }

    def expand_subgraph(self, node_ids, hops):
        all_nodes = set(node_ids)

        for _ in range(hops):
            query = """
            MATCH (n:Entity)-[r]-(m:Entity)
            WHERE n.id IN $node_ids
            RETURN DISTINCT m.id as node_id
            """
            result = self.neo4j_conn.query(query, {"node_ids": list(all_nodes)})
            new_nodes = {record["node_id"] for record in result}
            all_nodes.update(new_nodes)

        return list(all_nodes)

    def extract_context_from_neo4j(self, node_ids, scores):
        query = """
        MATCH (n:Entity)
        WHERE n.id IN $node_ids
        OPTIONAL MATCH (n)-[r]->(m:Entity)
        RETURN n, collect({target: m.name, relation: r.type}) as relations
        """

        result = self.neo4j_conn.query(query, {"node_ids": node_ids})

        context_parts = []
        for i, record in enumerate(result):
            node = record["n"]
            relations = record["relations"]
            score = scores[i] if i < len(scores) else 0.0

            context_parts.append(
                f"Entity: {node.get('name', node.get('id'))} "
                f"(Type: {node.get('type', 'unknown')}, "
                f"Similarity: {score:.3f})\n"
                f"Description: {node.get('description', 'No description')}"
            )

            if relations:
                rel_strings = [
                    f"{node.get('name')} ---{rel['relation']}---> {rel['target']}"
                    for rel in relations
                    if rel["target"]
                ]
                if rel_strings:
                    context_parts.append(f"Relations: {'; '.join(rel_strings)}")

        return "\n\n".join(context_parts)

    def extract_context_from_subgraph(self, subgraph, central_nodes, scores):
        context_parts = []

        for i, node_id in enumerate(central_nodes):
            if node_id in subgraph:
                node_data = subgraph.nodes[node_id]
                score = scores[i] if i < len(scores) else 0.0

                context_parts.append(
                    f"Entity: {node_data.get('name', node_id)} "
                    f"(Type: {node_data.get('type', 'unknown')}, "
                    f"Similarity: {score:.3f})\n"
                    f"Description: {node_data.get('description', 'No description')}"
                )

                relations = []
                for successor in subgraph.successors(node_id):
                    if subgraph.has_edge(node_id, successor):
                        edge_data = subgraph.edges[node_id, successor]
                        successor_name = subgraph.nodes[successor].get(
                            "name", successor
                        )
                        relations.append(
                            f"{node_data.get('name', node_id)} ---{edge_data.get('relation', 'related_to')}---> {successor_name}"
                        )
                if relations:
                    context_parts.append(f"Relations: {'; '.join(relations)}")

        return "\n\n".join(context_parts)


class GraphRAGProcessor:
    def __init__(self):
        self.kg_builder = KnowledgeGraphBuilder()
        self.node_embedder = NodeEmbedder()
        self.neo4j_conn = Neo4jConnection()
        self.retriever = GraphRAGRetriever(self.neo4j_conn)

    def process_documents_to_graph(self, file_paths):
        all_chunks = []

        for file_path in file_paths:
            file_extension = Path(file_path).suffix.lower()

            try:
                if file_extension == ".ipynb":
                    chunks = process_ipynb_file(file_path)
                elif file_extension in [".py", ".js", ".java", ".cpp"]:
                    chunks = process_code_file(file_path)
                elif file_extension == ".pdf":
                    chunks = process_pdf_file(file_path)
                else:
                    # for extensions not supported, treat as text file
                    chunks = process_text_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error process: {e}")

        print("Building knowledge graph...")
        graph = self.kg_builder.build_graph_from_chunks(all_chunks)

        print("Embedding all the nodes...")
        node_embeddings = self.node_embedder.embed_all_nodes(graph)

        print("Storing into Neo4j...")
        self.retriever.store_graph_to_neo4j(graph, node_embeddings)

        return {
            "chunks_processed": len(all_chunks),
            "nodes_created": len(graph.nodes),
            "edges_created": len(graph.edges),
        }

    def query_graph(self, query, k=10):
        if not self.retriever:
            return "Knowledge graph not built yet. Please process documents first."

        result = self.retriever.retrieve_relevant_subgraph(query, k=k)
        return result["context"]

    def close(self):
        self.neo4j_conn.close()
