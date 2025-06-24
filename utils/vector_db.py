import qdrant_client
from qdrant_client.http.models import VectorParams, PayloadSchemaType, Distance
from qdrant_client.models import PointStruct

def save_ipynb_files_embeddings_to_qdrant(chunks, collection_name="ipynb_embeddings"):
    client = qdrant_client.QdrantClient(url="http://localhost:6333")

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(chunks[0]['embedding']), distance="Cosine")
        )
    except Exception as e:
        # Collection already exists
        pass

    points = []
    for i, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=i,
                vector=chunk.get('embedding'),
                payload={
                    "original_id": chunk.get('id'),
                    "type": chunk.get('type'),
                    "content": chunk.get('content'),
                    "source_document": chunk.get('source_document'),
                    "cell_number": chunk.get('cell_number'),
                }
            )
        )
    if points:
        operation_info = client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points
        )
    return client

# def save_py_files_embeddings_to_qdrant(chunks, collection_name="py_embeddings"):






