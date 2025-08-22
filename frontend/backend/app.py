import logging
import os.path
import traceback

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from pipeline.orchestrator_agent_wise_feedback import pipeline
from flask_cors import CORS

from retrievers.graph_retriever import GraphRAGProcessor
from utils.vector_db import VectorProcessor

from dotenv import load_dotenv

from qdrant_client import QdrantClient
from neo4j import GraphDatabase
import urllib.parse

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

graph_processor = GraphRAGProcessor()

vector_processor = VectorProcessor()

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/apis/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
        user_query = data.get("user_query")
        if not user_query:
            return (
                jsonify(
                    {"status": "error", "message": "Missing user_query in request"}
                ),
                400,
            )

        logger.info(f"Processing query: {user_query}")
        result = pipeline(user_query)
        logger.info("Query processed successfully")
        return jsonify({"status": "success", "data": result["answers"]}), 200
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/apis/upload", methods=["POST"])
def file_upload():
    try:
        if "file" not in request.files:
            return jsonify({"status": "eror", "message": "No file provided"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"status": "error", "message": "No file selected"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        old_version_deleted = False
        if os.path.exists(filepath):
            logger.info(f"File {filename} already exists. Deleting old version...")
            try:
                # Delete from Qdrant
                qdrant_client = QdrantClient(host="localhost", port=6333)
                collections_info = qdrant_client.get_collections()
                qdrant_deleted_points = 0

                for collection in collections_info.collections:
                    try:
                        points = qdrant_client.scroll(
                            collection_name=collection.name,
                            limit=10000,
                            with_payload=True,
                        )[0]

                        points_to_delete = []
                        for point in points:
                            if point.payload:
                                source_document = point.payload.get(
                                    "source_document", ""
                                )
                                if source_document:
                                    extracted_file_name = os.path.basename(
                                        source_document
                                    )
                                    if extracted_file_name == filename:
                                        points_to_delete.append(point.id)

                        if points_to_delete:
                            qdrant_client.delete(
                                collection_name=collection.name,
                                points_selector=points_to_delete,
                            )
                            qdrant_deleted_points += len(points_to_delete)
                            logger.info(
                                f"Deleted {len(points_to_delete)} points from {collection.name}"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error deleting from collection {collection.name}: {e}"
                        )

                # Delete from Neo4j
                neo4j_deleted = 0
                try:
                    driver = GraphDatabase.driver(
                        "bolt://localhost:7687",
                        auth=("neo4j", os.environ.get("NEO4J_PASSWORD")),
                    )
                    with driver.session() as session:
                        result = session.run(
                            """
                        MATCH (n)
                        WHERE n.source_chunk CONTAINS $file_name OR n.source_document CONTAINS $file_name
                        WITH collect(n) AS nodes
                        UNWIND nodes AS n
                        OPTIONAL MATCH (n)-[r]-()
                        DELETE r, n
                        RETURN count(n) AS deleted_count
                        """,
                            file_name=filename,
                        )

                        record = result.single()
                        neo4j_deleted = record["deleted_count"] if record else 0
                        logger.info(f"Deleted {neo4j_deleted} nodes from Neo4j")

                        # Delete isolated entities
                        session.run("MATCH (e) WHERE NOT (e)--() DELETE e")
                    driver.close()
                except Exception as e:
                    logger.error(f"Error deleting from Neo4j: {e}")

                # Delete old file from disk
                os.remove(filepath)
                old_version_deleted = True
                logger.info(f"Deleted old version of {filename} from disk")

            except Exception as e:
                logger.error(f"Error deleting old version: {e}")

        # save file
        file.save(filepath)

        try:
            graph_result = graph_processor.process_documents_to_graph([filepath])
            vector_result = vector_processor.process_documents([filepath])

            response_data = {
                "status": "success",
                "message": "File uploaded and processed to both graph and vector databases",
                "filename": filename,
                "graph_stats": {
                    "nodes_created": graph_result.get("nodes_created", 0),
                    "edges_created": graph_result.get("edges_created", 0),
                },
                "vector_stats": {
                    "documents_added": vector_result.get("documents_added", 0),
                    "chunks_created": vector_result.get("chunks_created", 0),
                },
            }

            if old_version_deleted:
                response_data["message"] = (
                    "File updated - old version deleted and new version processed"
                )
                response_data["old_version_deleted"] = True

            return jsonify(response_data), 200

        except Exception as processing_error:
            if os.path.exists(filepath):
                os.remove(filepath)
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Processing failed: {str(processing_error)}",
                    }
                ),
                500,
            )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/apis/uploaded-files", methods=["GET"])
def get_uploaded_files():
    try:
        client = QdrantClient(host="localhost", port=6333)
        collections_info = client.get_collections()

        files_info = []
        for collection in collections_info.collections:
            try:
                points, next_page = client.scroll(
                    collection_name=collection.name, limit=1000, with_payload=True
                )

                if len(points) == 0:
                    continue

                unique_files = {}
                for point in points:
                    source_document = point.payload.get("source_document", "")
                    if source_document:
                        file_name = os.path.basename(source_document)
                    else:
                        file_name = "unknown_file"

                    if file_name not in unique_files:
                        file_extension = os.path.splitext(file_name)[1].lower()
                        file_type = (
                            file_extension.replace(".", "")
                            if file_extension
                            else "unknown"
                        )

                        unique_files[file_name] = {
                            "name": file_name,
                            "collection": collection.name,
                            "type": file_type,
                            "size": point.payload.get("file_size", 0),
                            "upload_date": point.payload.get("upload_date", ""),
                            "chunks_count": 0,
                        }
                    unique_files[file_name]["chunks_count"] += 1
                files_info.extend(list(unique_files.values()))
            except Exception as collection_error:
                print(
                    f"Error processing collection {collection.name}: {collection_error}"
                )
                continue
        return jsonify({"files": files_info, "total_files": len(files_info)})
    except Exception as e:
        print(f"Error in get_uploaded_files: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/apis/delete-uploaded-files/<file_name>", methods=["DELETE"])
def delete_uploaded_file(file_name):
    try:
        # decode file name
        file_name = urllib.parse.unquote(file_name)
        collection = request.json.get("collection") if request.json else None

        print(f"Deleting file: {file_name} from collection: {collection}")

        # 1. delete from qdrant
        qdrant_client = QdrantClient(host="localhost", port=6333)

        if collection:
            collections_to_check = [collection]
        else:
            collections_info = qdrant_client.get_collections()
            collections_to_check = [col.name for col in collections_info.collections]

        deleted_points = 0
        for col_name in collections_to_check:
            try:
                points = qdrant_client.scroll(
                    collection_name=col_name, limit=10000, with_payload=True
                )[0]

                points_to_delete = []
                for point in points:
                    if point.payload:
                        source_document = point.payload.get("source_document", "")
                        if source_document:
                            extracted_file_name = os.path.basename(source_document)
                            if extracted_file_name == file_name:
                                points_to_delete.append(point.id)

                if points_to_delete:
                    qdrant_client.delete(
                        collection_name=col_name, points_selector=points_to_delete
                    )
                    deleted_points += len(points_to_delete)
                    print(f"Deleted {len(points_to_delete)} points from {col_name}")

            except Exception as e:
                print(f"Error deleting from collection {col_name}: {e}")
                continue

        # 2. delete from neo4j
        neo4j_deleted = 0
        try:
            driver = GraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j", os.environ.get("NEO4J_PASSWORD")),
            )

            with driver.session() as session:
                # delete query, including entity and relationship
                result = session.run(
                    """
                MATCH (n)
                WHERE n.source_chunk CONTAINS $file_name OR n.source_document CONTAINS $file_name
                WITH collect(n) AS nodes
                UNWIND nodes AS n
                OPTIONAL MATCH (n)-[r]-()
                DELETE r, n
                RETURN count(n) AS deleted_count
                """,
                    file_name=file_name,
                )

                record = result.single()
                neo4j_deleted = record["deleted_count"] if record else 0
                print(f"Deleted {neo4j_deleted} document nodes from Neo4j")

                # delete isolated entity
                session.run(
                    """
                MATCH (e)
                WHERE NOT (e)--()
                DELETE e
                """
                )

            driver.close()

        except Exception as e:
            print(f"Error deleting from Neo4j: {e}")

        # delete local file
        file_deleted_from_disk = False
        try:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                file_deleted_from_disk = True
                print(f"Deleted file: {file_path}")
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error deleting file: {e}")

        return (
            jsonify(
                {
                    "message": f"File {file_name} deleted.",
                    "deleted_from_qdrant": deleted_points,
                    "deleted_from_neo4j": neo4j_deleted,
                    "deleted_from_disk": file_deleted_from_disk,
                }
            ),
            200,
        )

    except Exception as e:
        print(f"Error in delete_uploaded_file: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
