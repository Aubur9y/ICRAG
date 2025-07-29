from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.brain_agent import BrainAgent
from agents.decision_agent import DecisionAgent
from agents.decomposition_agent import DecompositionAgent
import logging
import re
import tiktoken
import time
import os
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from utils.reranker import Reranker
from agents.evaluator_agent_wise_feedback import ComponentEvaluator


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

    component_evaluator = ComponentEvaluator()

    decomposition_feedback = None
    retrieval_feedback = {}
    generation_feedback = None

    for iteration in range(max_retries + 1):
        logger.info(f"=== Pipeline Iteration {iteration + 1} ===")

        brain_agent = BrainAgent()
        decomposition_agent = DecompositionAgent()

        if decomposition_feedback and iteration > 0:
            decompose_prompt = f"""
            Based on previous feedback, improve the query decomposition:
            
            FEEDBACK: {decomposition_feedback}
            
            Original query: {user_query}
            """
            decompose_output = decomposition_agent.process_with_feedback(
                user_query, decompose_prompt
            )
        else:
            decompose_output = decomposition_agent.process(user_query)

        sub_tasks = decompose_output.get("sub_tasks", [])
        best_rewritten_query = decompose_output.get("best_rewritten_query")

        if not sub_tasks:
            sub_tasks = [best_rewritten_query or user_query]

        pipeline_results = []

        def process_sub_task_with_feedback(sub_task_query):
            task_feedback = retrieval_feedback.get(sub_task_query)
            if task_feedback and iteration > 0:
                retrieval_result = brain_agent.retrieve_with_feedback(
                    sub_task_query, task_feedback
                )
            else:
                retrieval_result = brain_agent.retrieve(sub_task_query)

            return {
                "sub_task_query": sub_task_query,
                "retrieval_details": retrieval_result,
            }

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_sub_task_with_feedback, q) for q in sub_tasks
            ]
            for future in as_completed(futures):
                pipeline_results.append(future.result())

        contexts_for_generation = []
        for result_item in pipeline_results:
            details = result_item.get("retrieval_details", {})
            if details.get("status") == "tool_executed":
                tool_output = details.get("tool_output")
                if tool_output:
                    contexts_for_generation.append(str(tool_output))
            elif details.get("status") == "direct_answer":
                direct_answer = details.get("direct_answer_content")
                if direct_answer:
                    contexts_for_generation.append(str(direct_answer))
        if not contexts_for_generation:
            return {
                "question": user_query,
                "contexts": [],
                "answers": "Sorry, I couldn't find any information related to your query.",
                "evaluation_scores": {"total": 0},
            }

        # reranker = Reranker()
        # if contexts_for_generation is None:
        #     contexts_for_generation = []
        # else:
        #     contexts_for_generation = reranker.rerank(
        #         best_rewritten_query, contexts_for_generation
        #     )

        final_context_str = "\n\n---\n\n".join(contexts_for_generation)
        decision_agent = DecisionAgent()

        if generation_feedback and iteration > 0:
            generation_prompt = f"""
            Based on previous feedback, improve the answer generation:

            FEEDBACK: {generation_feedback}

            Query: {best_rewritten_query or user_query}
            Context: {final_context_str}
            """
        else:
            generation_prompt = f"""
            Answer the question: "{best_rewritten_query or user_query}" based on the following context:
            {final_context_str}
            """

        generated_response = decision_agent.generate(generation_prompt)
        current_response_content = generated_response

        decomp_eval = component_evaluator.evaluate_decomposition(
            user_query, sub_tasks, contexts_for_generation
        )

        retrieval_evals = {}
        for result_item in pipeline_results:
            sub_task = result_item["sub_task_query"]
            details = result_item["retrieval_details"]
            ret_eval = component_evaluator.evaluate_retrieval(
                sub_task, details, "unknown"
            )
            retrieval_evals[sub_task] = ret_eval

        gen_eval = component_evaluator.evaluate_generation(
            best_rewritten_query or user_query,
            final_context_str,
            current_response_content,
        )

        total_score = (
            decomp_eval["score"]
            + sum(eval_data["score"] for eval_data in retrieval_evals.values())
            / len(retrieval_evals)
            + gen_eval["score"]
        ) / 3

        logger.info(f"Iteration {iteration + 1} scores:")
        logger.info(f"Decomposition: {decomp_eval['score']:.2f}")
        logger.info(
            f"Retrieval avg: {sum(eval_data['score'] for eval_data in retrieval_evals.values()) / len(retrieval_evals):.2f}"
        )
        logger.info(f"Generation: {gen_eval['score']:.2f}")
        logger.info(f"Total: {total_score:.2f}")

        if total_score >= 8.0 or iteration >= max_retries:
            break

        if decomp_eval["score"] < 7.0:
            decomposition_feedback = decomp_eval["feedback"]

        for sub_task, eval_data in retrieval_evals.items():
            if eval_data["score"] < 7.0:
                retrieval_feedback[sub_task] = eval_data["feedback"]

        if gen_eval["score"] < 7.0:
            generation_feedback = gen_eval["feedback"]

    # if not contexts_for_generation:
    #     return {
    #         "question": user_query,
    #         "contexts": [],
    #         "answers": "Sorry, I couldn't find any information related to your query.",
    #         "evaluation_scores": {"total": 0},
    #     }

    logger.info(
        f"Pipeline returning: question={user_query[:50]}, contexts_count={len(contexts_for_generation)}, answers_length={len(current_response_content)}"
    )
    return {
        "question": user_query,
        "contexts": contexts_for_generation,
        "answers": current_response_content,
        "evaluation_scores": {
            "decomposition": decomp_eval["score"],
            "retrieval": {k: v["score"] for k, v in retrieval_evals.items()},
            "generation": gen_eval["score"],
            "total": total_score,
        },
    }


if __name__ == "__main__":
    query = "Can you summarize the tasks needed to do the ads project regarding the flood risk, in the steps needed, also what is the technical requirements for each task?"

    results = pipeline(query)

    print("\n" + "=" * 25 + " Final Generated Answer " + "=" * 25 + "\n")
    print(results)
    print("\n" + "=" * 72)
