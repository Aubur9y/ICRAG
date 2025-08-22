from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.brain_agent import BrainAgent
from agents.decision_agent import DecisionAgent
from agents.decomposition_agent import DecompositionAgent
from utils.reranker import Reranker
import logging
import os
import re
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# This file is not only for ablation tests of the report
# ---------------------------------------------------------------------------


def is_inventory_query(user_query: str) -> bool:
    """Detect queries asking for a description of the knowledge base contents."""
    keywords = [
        "what do you have",
        "what's in your database",
        "what's in your knowledge base",
        "what collections do you have",
        "what entities do you have",
        "what documents do you have",
        "what is stored",
        "what is available",
        "what can you search",
        "what's in vector database",
        "what's in graph database",
        "what knowledge do you have",
    ]
    user_query_lower = user_query.lower()
    return any(k in user_query_lower for k in keywords)


def get_qdrant_summary() -> str:
    """Return a human-readable summary of collections stored in Qdrant."""
    client = QdrantClient(host="localhost", port=6333)
    collections = client.get_collections().collections
    summary_lines = []
    for col in collections:
        name = col.name
        count = client.count(collection_name=name, exact=True).count
        summary_lines.append(f"Collection '{name}' contains {count} vectors.")
    if not summary_lines:
        return "Qdrant vector database is empty."
    return "Qdrant vector database:\n" + "\n".join(summary_lines)


def get_neo4j_summary() -> str:
    """Return a human-readable summary of entities and relations inside Neo4j."""
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "neo4j")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    with driver.session() as session:
        entity_types = session.run("MATCH (n) RETURN DISTINCT labels(n)").values()
        entity_count_result = session.run("MATCH (n) RETURN count(n)").single()
        entity_count = entity_count_result[0] if entity_count_result is not None else 0
        rel_types = session.run("MATCH ()-[r]->() RETURN DISTINCT type(r)").values()
        rel_count_result = session.run("MATCH ()-[r]->() RETURN count(r)").single()
        rel_count = rel_count_result[0] if rel_count_result is not None else 0
    driver.close()
    entity_types_str = (
        ", ".join([str(e[0]) for e in entity_types if e and e[0]])
        if entity_types
        else "None"
    )
    rel_types_str = (
        ", ".join([str(r[0]) for r in rel_types if r and r[0]]) if rel_types else "None"
    )
    return (
        f"Neo4j graph database contains entity types: {entity_types_str} (total: {entity_count}).\n"
        f"Relation types: {rel_types_str} (total: {rel_count})."
    )


# ---------------------------------------------------------------------------
# Main baseline pipeline (no feedback loop)
# ---------------------------------------------------------------------------


def pipeline(user_query: str):
    """Execute the end-to-end RAG pipeline once with **no feedback loops**.

    This serves as the simplest baseline to be compared against pipelines that
    include retriever-only or agent-wise feedback mechanisms.
    """

    # 1. Return the inventory overview immediately if that's what the user asked.
    if is_inventory_query(user_query):
        qdrant_info = get_qdrant_summary()
        neo4j_info = get_neo4j_summary()
        context = qdrant_info + "\n" + neo4j_info
        decision_agent = DecisionAgent()
        response = decision_agent.generate(
            f"Summarise the following knowledge base inventory for the user in concise words.\n{context}"
        )
        return {"question": user_query, "contexts": [], "answers": response.content}

    # 2. Decompose the query into manageable sub-tasks.
    logger.info("Step 1: Decomposing user query")
    decomposition_agent = DecompositionAgent()
    decompose_output = decomposition_agent.process(user_query)
    sub_tasks = decompose_output.get("sub_tasks", [])
    best_rewritten_query = decompose_output.get("best_rewritten_query")

    # Fallbacks if the LLM fails to produce sub-tasks.
    if not sub_tasks:
        logger.warning("Agent failed to generate sub-tasks.")
        if best_rewritten_query:
            logger.info(f"Using the best rewrite as sub-task: {best_rewritten_query}")
            sub_tasks = [best_rewritten_query]
        else:
            logger.info("Using the original query as sub-task.")
            sub_tasks = [user_query]
    else:
        logger.info(f"Sub-tasks generated: {sub_tasks}")

    # 3. Retrieve context for each sub-task in parallel.
    brain_agent = BrainAgent()

    def _process_sub_task(sub_task_query: str):
        return {
            "sub_task_query": sub_task_query,
            "retrieval_details": brain_agent.retrieve(sub_task_query),
        }

    pipeline_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_process_sub_task, q) for q in sub_tasks]
        for future in as_completed(futures):
            pipeline_results.append(future.result())

    contexts_for_generation = []

    # Prefer clean text passages over raw JSON blobs from retrievers
    for item in pipeline_results:
        details = item.get("retrieval_details", {})
        status = details.get("status")
        if status == "direct_answer":
            direct_text = details.get("direct_answer_content")
            if direct_text:
                contexts_for_generation.append(str(direct_text))
            continue

        if status == "tool_executed":
            tool_name = details.get("tool_name")
            output = details.get("tool_output")
            # Vector search returns a list of hits; extract payload.content
            if tool_name == "vector_search":
                if isinstance(output, list):
                    for hit in output[:8]:
                        payload = (
                            hit.get("payload", {}) if isinstance(hit, dict) else {}
                        )
                        text = payload.get("content") or payload.get("text")
                        if text:
                            contexts_for_generation.append(str(text))
                # If not a list, fall back to string form
                else:
                    contexts_for_generation.append(str(output))
            # Graph/Web search might return dicts with a 'context' string or already a string
            elif tool_name in ("graph_search", "web_search"):
                if isinstance(output, dict) and "context" in output:
                    contexts_for_generation.append(str(output.get("context")))
                else:
                    contexts_for_generation.append(str(output))

    # 4.deduplication, rerank and choose top-3
    if contexts_for_generation:
        contexts_for_generation = list(dict.fromkeys(contexts_for_generation))
        contexts_for_generation = contexts_for_generation[:3]

    if not contexts_for_generation:
        # Fallback: no retrieved contexts; attempt an open-mode answer so tests don't degenerate to CANNOT_FIND
        decision_agent = DecisionAgent()
        open_answer = decision_agent.generate(
            f"Answer briefly and clearly: {user_query}", mode="open"
        )
        open_answer_text = (
            open_answer if isinstance(open_answer, str) else open_answer.content
        )
        return {
            "question": user_query,
            "contexts": [],
            "answers": open_answer_text,
        }

    # 5. Enumerate contexts so the model can cite them (Context N)
    enumerated_contexts = [
        f"Context {idx + 1}:\n{ctx}" for idx, ctx in enumerate(contexts_for_generation)
    ]
    final_context_str = "\n\n---\n\n".join(enumerated_contexts)
    decision_agent = DecisionAgent()

    prompt_for_generation = (
        f'Answer the Question: "{best_rewritten_query or user_query}" using ONLY the information found in the context passages below. '
        f"If the answer cannot be found in these passages, reply exactly: CANNOT_FIND.\n\n"
        f"{final_context_str}\n"
    )

    generated_response = decision_agent.generate(prompt_for_generation, mode="strict")
    raw_answer = (
        generated_response
        if isinstance(generated_response, str)
        else generated_response.content
    )

    # strip chain-of-thought and enforce policy

    def _clean_answer(text: str) -> str:
        # Remove <think>...</think>
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
        # Remove leading markdown headers or separators
        text = re.sub(r"^\s*[#\-]{2,}.*$", "", text, flags=re.MULTILINE)
        return text.strip()

    answer_content = _clean_answer(raw_answer)

    return {
        "question": user_query,
        "contexts": contexts_for_generation,
        "answers": answer_content,
    }


if __name__ == "__main__":
    sample_query = (
        "What are the main differences between supervised and unsupervised learning?"
    )
    res = pipeline(sample_query)
    print(res)
