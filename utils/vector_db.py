import uuid
import time
import qdrant_client
from qdrant_client.http.models import VectorParams
from qdrant_client.models import PointStruct
from pathlib import Path
from utils.file_embedding import (
    process_ipynb_file,
    process_code_file,
    process_pdf_file,
    process_text_file,
)
from utils.constants import supported_code_file_extensions


def qdrant_connection(url="http://localhost:6333"):
    return qdrant_client.QdrantClient(url=url)


def generate_unique_id():
    return int(time.time() * 1000000) + hash(uuid.uuid4()) % 1000000


def save_ipynb_files_embeddings_to_qdrant(
    file_path, collection_name="ipynb_embeddings"
):
    client = qdrant_connection()

    chunks = process_ipynb_file(file_path)

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(chunks[0]["embedding"]), distance="Cosine"
            ),
        )
    except Exception as e:
        # Collection already exists
        pass

    # Store into vector db
    points = []
    for i, chunk in enumerate(chunks):
        points.append(
            # every entry is a point, pointstruct is to give a clear structure
            PointStruct(
                id=generate_unique_id(),
                vector=chunk.get("embedding"),
                payload={
                    "original_id": chunk.get("id"),
                    "type": chunk.get("type"),
                    "content": chunk.get("content"),
                    "source_document": chunk.get("source_document"),
                    "cell_number": chunk.get("cell_number"),
                },
            )
        )
    if points:
        operation_info = client.upsert(
            collection_name=collection_name, wait=True, points=points
        )
    return client


def save_code_files_embeddings_to_qdrant(file_path, collection_name="code_embeddings"):
    client = qdrant_connection()

    chunks = process_code_file(file_path)
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(chunks[0]["embedding"]), distance="Cosine"
            ),
        )
    except Exception as e:
        pass

    points = []
    for i, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=generate_unique_id(),
                vector=chunk.get("embedding"),
                payload={
                    "original_id": chunk.get("id"),
                    "type": "code",
                    "content": chunk.get("content"),
                    "source_document": chunk.get("source_document"),
                    "chunk_number": chunk.get("chunk_number"),
                    "language": chunk.get("language"),
                    "block_name": chunk.get("block_name"),
                    "start_line": chunk.get("start_line"),
                    "end_line": chunk.get("end_line"),
                },
            )
        )
    if points:
        operation_info = client.upsert(
            collection_name=collection_name, wait=True, points=points
        )
    return client


def save_pdf_files_embeddings_to_qdrant(file_path, collection_name="pdf_embeddings"):
    client = qdrant_connection()
    chunks = process_pdf_file(file_path)
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(chunks[0]["embedding"]), distance="Cosine"
            ),
        )
    except Exception as e:
        pass

    points = []
    for i, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=generate_unique_id(),
                vector=chunk.get("embedding"),
                payload={
                    "original_id": chunk.get("id"),
                    "type": "pdf",
                    "content": chunk.get("content"),
                    "source_document": chunk.get("source_document"),
                    "chunk_number": chunk.get("chunk_number"),
                },
            )
        )
    if points:
        operation_info = client.upsert(
            collection_name=collection_name, wait=True, points=points
        )
    return client


def save_text_files_embeddings_to_qdrant(file_path, collection_name="text_embeddings"):
    client = qdrant_connection()
    chunks = process_text_file(file_path)
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(chunks[0]["embedding"]), distance="Cosine"
            ),
        )
    except Exception as e:
        pass

    points = []
    for i, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=generate_unique_id(),
                vector=chunk.get("embedding"),
                payload={
                    "original_id": chunk.get("id"),
                    "type": "text",
                    "content": chunk.get("content"),
                    "source_document": chunk.get("source_document"),
                    "chunk_number": chunk.get("chunk_number"),
                },
            )
        )
    if points:
        operation_info = client.upsert(
            collection_name=collection_name, wait=True, points=points
        )
    return client


class VectorProcessor:
    def __init__(self):
        self.client = qdrant_connection()

    def process_documents(self, file_paths):
        total_chunks = 0
        processed_files = 0

        for file_path in file_paths:
            try:
                file_extension = Path(file_path).suffix.lower()

                if file_extension == ".ipynb":
                    save_ipynb_files_embeddings_to_qdrant(file_path)
                    chunks = process_ipynb_file(file_path)
                elif file_extension in supported_code_file_extensions:
                    save_code_files_embeddings_to_qdrant(file_path)
                    chunks = process_code_file(file_path)
                elif file_extension == ".pdf":
                    save_pdf_files_embeddings_to_qdrant(file_path)
                    chunks = process_pdf_file(file_path)
                else:
                    save_text_files_embeddings_to_qdrant(file_path)
                    chunks = process_text_file(file_path)

                total_chunks += len(chunks)
                processed_files += 1

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        return {"documents_added": processed_files, "chunks_created": total_chunks}


if __name__ == "__main__":
    file_path = "../documents/macm2.txt"
    result = save_text_files_embeddings_to_qdrant(file_path)
    print(result)
