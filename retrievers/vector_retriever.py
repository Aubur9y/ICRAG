from qdrant_client import QdrantClient
from utils.ollama_client import OllamaClient

class VectorRetrieverAgent:
    def __init__(
            self,
            collection_name,
            qdrant_host="localhost",
            qdrant_port=6333,
            embedding_model_name="mxbai-embed-large",
            top_k=5
    ):
        self.collection_name = collection_name
        self.top_k = top_k

        self.embedding_model = embedding_model_name

        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

    def embed_query(self, query):
        ollama_client = OllamaClient()
        response = ollama_client.embed(model=self.embedding_model, input_text=query)
        return response['embeddings'][0]

    def retrieve(self, query, filter=None,similarity_threshold=0.7):
        vector = self.embed_query(query)

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=self.top_k,
            query_filter=filter,
            with_payload=True,
        )

        results = []
        if search_result:
            for point in search_result:
                if point.score >= similarity_threshold:
                    results.append({
                        "id": point.id,
                        "score": point.score,
                        "payload": point.payload
                    })
        if not results:
            return {
                "status": "no_relevant_information_found"
            }
        return results


if __name__ == "__main__":
    agent = VectorRetrieverAgent(collection_name="ipynb_embeddings")
    results = agent.retrieve("After the presentations, what to do?")
    for r in results:
        print(f"\nScore: {r['score']}\nContent: {r['payload'].get('content')}")

