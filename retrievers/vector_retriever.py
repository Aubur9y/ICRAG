import os
import logging
from qdrant_client import QdrantClient
from utils.ollama_client import OllamaClient
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


class VectorRetrieverAgent:
    def __init__(
        self,
        collections=None,
        qdrant_host="localhost",
        qdrant_port=6333,
        embedding_model_name="mxbai-embed-large",
        top_k=5,
    ):
        if collections is None:
            collections = [
                "code_embeddings",
                "ipynb_embeddings",
                "pdf_embeddings",
                "text_embeddings",
            ]
        self.collections = collections
        self.top_k = top_k

        self.embedding_model = embedding_model_name

        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

    def embed_query(self, query):
        ollama_client = OllamaClient()
        response = ollama_client.embed(model=self.embedding_model, input_text=query)
        return response["embeddings"][0]

    def retrieve(self, query, filter=None, similarity_threshold=0.7):
        vector = self.embed_query(query)

        search_results = []

        # check collection existance
        existing_collections = []
        collections = self.client.get_collections()
        for collection in self.collections:
            if any(col.name == collection for col in collections.collections):
                existing_collections.append(collection)
        if not existing_collections:
            return {
                "status": "no_relevant_information_found",
                "message": "No collections available for search.",
            }

        for collection in existing_collections:
            result = self.client.search(
                collection_name=collection,
                query_vector=vector,
                limit=self.top_k,
                query_filter=filter,
                with_payload=True,
            )
            if result:
                search_results.extend(result)

        search_results = sorted(search_results, key=lambda x: x.score, reverse=True)[
            : self.top_k
        ]

        # logger.info(f"Search results from vector retriever: {search_results}")

        results = []
        if search_results:
            for point in search_results:
                if point.score >= similarity_threshold:
                    results.append(
                        {"id": point.id, "score": point.score, "payload": point.payload}
                    )
        if not results:
            return {"status": "no_relevant_information_found"}
        return results

    def clear_all_databases(self):
        try:
            collections_response = self.client.get_collections()
            collections = [
                collection.name for collection in collections_response.collections
            ]

            for collection in collections:
                self.client.delete_collection(collection_name=collection)

            print("All vector dbs cleaned")

        except Exception as e:
            print(e)

        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j", os.environ.get("NEO4J_PASSWORD")),
            )

            with driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                print("Graph db cleaned")

            driver.close()

        except Exception as e:
            print(e)


if __name__ == "__main__":
    agent = VectorRetrieverAgent()
    agent.clear_all_databases()
