# ========================
# This file is not used in the final product, only in the early stages for exploration.
# ========================

import qdrant_client


def check_qdrant_collection_data(collection_name="ipynb_embeddings"):
    client = qdrant_client.QdrantClient(url="http://localhost:6333")

    try:
        points, next_page_offset = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            print(f"ID: {point.id}, Payload: {point.payload}")
    except Exception as e:
        print(e)


check_qdrant_collection_data()
