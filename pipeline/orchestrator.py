from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.brain_agent import BrainAgent
from agents.decision_agent import DecisionAgent
from agents.decomposition_agent import DecompositionAgent
from agents.evaluator import Evaluator
import logging
import re
import tiktoken
import time
import os
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from utils.reranker import Reranker

# ---------------------------------------------------------------------------
# This file is not only for ablation tests of the report
# ---------------------------------------------------------------------------


class ToolReturnFilter(logging.Filter):
    def __init__(self, name=""):
        super().__init__(name)
        self.pattern = re.compile(r"^Tool '.*' returns:")

    def filter(self, record):
        return not self.pattern.match(record.getMessage())


logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)

brain_agent_logger = logging.getLogger("agents.brain_agent")
brain_agent_logger.addFilter(ToolReturnFilter())

logger = logging.getLogger(__name__)


def truncate_text_by_tokens(text, model_name, max_tokens):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens, errors="ignore")
    return text


def is_inventory_query(user_query):
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


def get_qdrant_summary():
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


def get_neo4j_summary():
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


def pipeline(user_query, max_retries=2):
    t0 = time.time()
    # Inventory query shortcut
    if is_inventory_query(user_query):
        qdrant_info = get_qdrant_summary()
        neo4j_info = get_neo4j_summary()
        context = qdrant_info + "\n" + neo4j_info
        decision_agent = DecisionAgent()
        response = decision_agent.generate(
            f"Summarise the following knowledge base inventory for the user in concise words.\n{context}"
        )
        return {"question": user_query, "contexts": [], "answers": response.content}

    brain_agent = BrainAgent()
    decomposition_agent = DecompositionAgent()

    # query decomposition
    logger.info("Step 1: Decomposing user query")
    decompose_output = decomposition_agent.process(user_query)
    sub_tasks = decompose_output.get("sub_tasks", [])
    best_rewritten_query = decompose_output.get("best_rewritten_query")

    # logic for if there is no sub-tasks
    if not sub_tasks:
        logger.warning("Agent failed to generate sub-tasks.")
        if best_rewritten_query:
            logger.info(f"Using the best rewrite as sub-task: {best_rewritten_query}")
            sub_tasks = [best_rewritten_query]
        else:
            logger.info(f"Using the original query as sub-task: {user_query}")
            sub_tasks = [user_query]
    else:
        logger.info(f"Sub-tasks generated: {sub_tasks}")

    t1 = time.time()
    pipeline_results = []

    # Process sub-tasks concurrently
    def process_sub_task(sub_task_query):
        return {
            "sub_task_query": sub_task_query,
            "retrieval_details": brain_agent.retrieve(sub_task_query),
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_sub_task, q) for q in sub_tasks]
        for future in as_completed(futures):
            pipeline_results.append(future.result())

    logger.info(f"Time taken: {time.time() - t1:.2f}s")

    # prepare context for generation
    logger.info("Step 3: Aggregating context from sub-tasks.")
    contexts_for_generation = []

    # extract pure output
    for result_item in pipeline_results:
        details = result_item.get("retrieval_details", {})
        status = details.get("status")
        if status == "direct_answer":
            direct_text = details.get("direct_answer_content")
            if direct_text:
                contexts_for_generation.append(str(direct_text))
            continue

        if status == "tool_executed":
            tool_name = details.get("tool_name")
            output = details.get("tool_output")
            # extract payload.content from vector search
            if tool_name == "vector_search":
                if isinstance(output, list):
                    for item in output[:12]:
                        payload = (
                            item.get("payload", {}) if isinstance(item, dict) else {}
                        )
                        content = payload.get("content") or payload.get("text")
                        if content:
                            contexts_for_generation.append(str(content))
                else:
                    contexts_for_generation.append(str(output))
            # Graph/Web search prioritise context
            elif tool_name in ("graph_search", "web_search"):
                if isinstance(output, dict) and "context" in output:
                    contexts_for_generation.append(str(output.get("context")))
                else:
                    contexts_for_generation.append(str(output))

    has_relevant_info = any(
        result_item.get("retrieval_details", {}).get("status")
        not in ["no_relevant_information", "pipeline_error"]
        for result_item in pipeline_results
    )

    if not has_relevant_info or not contexts_for_generation:
        # return "Sorry, I couldn't find any information related to your query in my knowledge base."
        return {
            "question": user_query,
            "contexts": [],
            "answers": "Sorry, I couldn't find any information related to your query in my knowledge base.",
        }

    contexts_for_generation = list(dict.fromkeys(contexts_for_generation))
    reranker = Reranker()
    contexts_for_generation = reranker.rerank(
        best_rewritten_query or user_query, contexts_for_generation
    )[:8]

    # enumerate context, easy for model to cite by Context N
    enumerated_contexts = [
        f"Context {idx + 1}:\n{ctx}" for idx, ctx in enumerate(contexts_for_generation)
    ]
    final_context_str = "\n\n---\n\n".join(enumerated_contexts)

    decision_agent = DecisionAgent()
    current_response_content = ""
    last_feedback = None

    for attempt in range(max_retries + 1):
        logger.info(
            f"Step 4.{attempt + 1}: Generation attempt {attempt + 1}/{max_retries + 1}"
        )

        if attempt == 0:
            query_to_answer = (
                best_rewritten_query if best_rewritten_query else user_query
            )
            prompt_for_generation = f"""
            You will receive several context passages numbered as Context 1, Context 2, ... .

            Instructions:
            1. Answer the Question directly and concisely (≤150 words) using ONLY the information from those passages; cite them like (Context N).
            2. If the answer cannot be found in the passages, reply exactly: CANNOT_FIND.

            Question:
            {query_to_answer}

            Context passages:
            {final_context_str}
            """
        else:
            # calibration prompt
            logger.info("Constructing calibration prompt with feedback.")
            prompt_for_generation = f"""
            Your previous attempt to answer the question "{best_rewritten_query or user_query}" was not sufficient.

            Here is the feedback on your last answer:
            --- FEEDBACK ---
            {last_feedback}
            --- END FEEDBACK ---

            Here is the original context again (numbered as Context N):
            --- CONTEXT ---
            {final_context_str}
            --- END CONTEXT ---

            Please generate a new, improved answer that directly addresses the feedback and better utilises the provided context. Remember to cite like (Context N) and reply exactly CANNOT_FIND if not answerable from context.
            """
            logger.info(f"Feedback: {prompt_for_generation}")

        t_gen = time.time()
        generated_response = decision_agent.generate(prompt_for_generation)
        raw_answer_text = (
            generated_response
            if isinstance(generated_response, str)
            else generated_response.content
        )

        # Strip any chain-of-thought blocks like <think>...</think> or leading markdown headers
        import re

        def _clean_answer(text: str) -> str:
            text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^\s*[#\-]{2,}.*$", "", text, flags=re.MULTILINE)
            return text.strip()

        current_response_content = _clean_answer(raw_answer_text)
        logger.info(f"Time taken for generation: {time.time() - t_gen:.2f}s")

        logger.info(f"Step 5.{attempt + 1}: Evaluating the generated response.")

        EVALUATOR_MODEL = "gpt-3.5-turbo"
        MAX_CONTEXT_TOKENS_FOR_EVALUATOR = 12000

        truncated_context_for_eval = truncate_text_by_tokens(
            final_context_str, EVALUATOR_MODEL, MAX_CONTEXT_TOKENS_FOR_EVALUATOR
        )

        evaluator = Evaluator(
            best_rewritten_query, truncated_context_for_eval, current_response_content
        )
        evaluation_result = evaluator.evaluate()

        logger.info(
            f"Attempt {attempt + 1} evaluation score: {evaluation_result['score']:.2f}"
        )

        if not evaluation_result["needs_improvement"]:
            logger.info("Generated response meets quality standards.")
            break
        else:
            last_feedback = "\n".join(evaluation_result["feedback"])
            if attempt < max_retries:
                logger.info(
                    f"Response quality is low. Retrying... ({attempt + 1}/{max_retries})"
                )
            else:
                logger.info(
                    "Max retries reached. Returning the last generated response."
                )

    logger.info(f"Final response: {current_response_content}")
    logger.info(f"Time taken for the whole pipeline: {time.time() - t0:.2f}s.")
    return {
        "question": user_query,
        "contexts": contexts_for_generation,
        "answers": current_response_content,
    }


if __name__ == "__main__":
    query = "Can you summarize the tasks needed to do the ads project regarding the flood risk, in the steps needed, also what is the technical requirements for each task?"

    results = pipeline(query)

    print("\n" + "=" * 25 + " Final Generated Answer " + "=" * 25 + "\n")
    print(results)
    print("\n" + "=" * 72)
