import json
import PyPDF2
from dotenv import load_dotenv
from utils.ollama_client import OllamaClient
from pathlib import Path
from openai import OpenAI

load_dotenv()

current_path = Path(__name__).resolve()
root = current_path.parent.parent
documents = root / "documents"


def embed_text(text_chunk, model="mxbai-embed-large"):
    ollama_client = OllamaClient()
    embedding = ollama_client.embed(model, input_text=text_chunk)
    return embedding["embeddings"][0]


def embed_code(code_chunk, model="mxbai-embed-large"):
    ollama_client = OllamaClient()
    embedding = ollama_client.embed(model, input_text=code_chunk)
    return embedding["embeddings"][0]


def process_ipynb_file(file_path):
    processed_chunks = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            notebook_content = json.load(f)
    except FileNotFoundError:
        print("File not found.")
        return processed_chunks
    except json.JSONDecodeError:
        print(f"Error: file {file_path} is not a valid JSON format.")
        return processed_chunks
    except Exception as e:
        print(f"Error reading document: {e}")
        return processed_chunks

    if "cells" not in notebook_content or not isinstance(
        notebook_content["cells"], list
    ):
        print(f"{file_path} is not a valid format")
        return processed_chunks

    for i, cell in enumerate(notebook_content["cells"]):
        cell_type = cell.get("cell_type")
        source_line = cell.get("source", [])

        if isinstance(source_line, list):
            content = "".join(source_line)
        else:
            content = source_line if source_line else ""
        if not content.strip():
            continue

        if cell_type == "markdown":
            embedding = embed_text(content)
            processed_chunks.append(
                {
                    "id": f"{file_path}_cell_{i}_md",
                    "type": "markdown",
                    "content": content,
                    "embedding": embedding,
                    "source_document": file_path,
                    "cell_number": i,
                }
            )
        elif cell_type == "code":
            embedding = embed_code(content)
            processed_chunks.append(
                {
                    "id": f"{file_path}_cell_{i}_code",
                    "type": "code",
                    "content": content,
                    "embedding": embedding,
                    "source_document": file_path,
                    "cell_number": i,
                }
            )
    return processed_chunks


def process_code_file(file_path, chunk_size=50):
    processed_chunks = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return processed_chunks
    except Exception as e:
        print(f"Error reading the file: {e}")
        return processed_chunks

    file_extension = Path(file_path).suffix.lower()[1:]
    chunks = split_code_into_chunks(content, file_extension, chunk_size)

    for i, chunk in enumerate(chunks):
        if not chunk["content"].strip():
            continue

        embedding = embed_code(chunk["content"])
        processed_chunks.append(
            {
                "id": f"{file_path}_chunk_{i}",
                "type": "code",
                "content": chunk["content"],
                "embedding": embedding,
                "source_document": str(file_path),
                "chunk_number": i,
                "language": file_extension,
                "block_name": chunk["block_name"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
            }
        )

    return processed_chunks


# Helper function for code processing function
def split_code_into_chunks(content, file_extension, chunk_size=50):
    # ChatGPT is used here to help deciding on how to chunk different code files
    lines = content.split("\n")
    chunks = []
    current_chunk = []
    current_block_name = None
    block_indent_level = 0
    block_start_line = 0

    for i, line in enumerate(lines):
        line_content = line.strip()

        # determine the start of the code block
        if file_extension == "py":
            if line_content.startswith(("def ", "class ")) and not line.startswith(" "):
                if current_chunk and current_block_name:
                    chunks.append(
                        {
                            "content": "\n".join(current_chunk),
                            "block_name": current_block_name,
                            "start_line": block_start_line,
                            "end_line": i - 1,
                        }
                    )
                # new code block
                current_chunk = [line]
                current_block_name = line_content.split("(")[0].split(":")[0].strip()
                block_indent_level = len(line) - len(line.strip())
                block_start_line = i
                continue

            # determine the end of a code block by intendation
            if (
                current_block_name
                and line
                and len(line) - len(line.lstrip()) <= block_indent_level
            ):
                chunks.append(
                    {
                        "content": "\n".join(current_chunk),
                        "block_name": current_block_name,
                        "start_line": block_start_line,
                        "end_line": i - 1,
                    }
                )
                current_chunk = [line]
                current_block_name = None
                block_start_line = i
                continue

        elif file_extension in ["java", "cpp", "js"]:
            if (
                line_content.startswith(
                    ("public", "private", "protected", "function", "class")
                )
                and "{" in line_content
            ):
                if current_chunk:
                    chunks.append(
                        {
                            "content": "\n".join(current_chunk),
                            "block_name": current_block_name or "code_segment",
                            "start_line": block_start_line,
                            "end_line": i - 1,
                        }
                    )

                current_chunk = [line]
                # extract function/class name
                if "class" in line_content:
                    current_block_name = (
                        line_content.split("class ")[1].split("{")[0].strip()
                    )
                else:
                    parts = line_content.split("(")[0].strip().split(" ")
                    current_block_name = parts[-1]  # extract function name

                block_start_line = i
                continue

        # add current line to current chunk
        current_chunk.append(line)

        # for none code parts, chunk with fixed size
        if not current_block_name and len(current_chunk) >= chunk_size:
            chunks.append(
                {
                    "content": "\n".join(current_chunk),
                    "block_name": "code_segment",
                    "start_line": block_start_line,
                    "end_line": i,
                }
            )
            current_chunk = []
            block_start_line = i + 1

    # add last chunk
    if current_chunk:
        chunks.append(
            {
                "content": "\n".join(current_chunk),
                "block_name": current_block_name or "code_segment",
                "start_line": block_start_line,
                "end_line": len(lines) - 1,
            }
        )

    return chunks


def process_text_file(file_path, chunk_size=300, overlap=50):
    processed_chunks = []
    try:
        with open(file_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return processed_chunks

    chunks = []
    for i in range(0, len(content), chunk_size - overlap):
        chunk = content[i : i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)

    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        processed_chunks.append(
            {
                "id": f"{file_path}_chunk_{i}",
                "type": "text",
                "content": chunk,
                "embedding": embedding,
                "source_document": file_path,
                "chunk_number": i,
            }
        )

    return processed_chunks


def process_pdf_file(file_path, chunk_size=500):
    processed_chunks = []
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            content = ""
            for page in reader.pages:
                content += page.extract_text() + "\n"
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return processed_chunks

    chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        embedding = embed_text(chunk)
        processed_chunks.append(
            {
                "id": f"{file_path}_chunk_{i}",
                "type": "text",
                "content": chunk,
                "embedding": embedding,
                "source_document": file_path,
                "chunk_number": i,
            }
        )

    return processed_chunks


if __name__ == "__main__":
    # ChatGPT is used to generate tests for this file
    print("\n===== Test code chunking =====")
    py_code_path = Path(__file__)

    with open(py_code_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks_raw = split_code_into_chunks(content, "py")
    print(f"Chunking result: {len(chunks_raw)} code chunks")

    for i, chunk in enumerate(chunks_raw):
        print(f"\nChunk {i + 1}: {chunk['block_name']}")
        print(f"Line range: {chunk['start_line']} - {chunk['end_line']}")
        print(f"Content length: {len(chunk['content'].split())} words")

        content_lines = [line for line in chunk["content"].split("\n") if line.strip()]
        preview_lines = content_lines[:10]
        preview = "\n".join(preview_lines)
        print(f"Content preview: \n{preview}...")

    print("\n===== Test code embedding =====")
    if chunks_raw:
        for i, chunk in enumerate(chunks_raw):
            content_words = len(chunk["content"].split())
            print(
                f"Embedding chunk {i}: '{chunk['block_name']}' ({content_words} words)..."
            )
            try:
                content_to_embed = " ".join(chunk["content"].split())
                embedding = embed_code(content_to_embed)
                print(f"Embedding dimensions: {len(embedding)}")
                print(f"First three values of the embedding: {embedding[:3]}")
            except Exception as e:
                print(f"Error: {e}")
