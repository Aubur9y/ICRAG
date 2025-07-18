import sys
import numpy as np
import re
import logging
from utils.ollama_client import OllamaClient
from utils.prompt_templates import (
    REWRITE_QUERY_SYSTEM_PROMPT,
    DECOMPOSE_QUERY_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


def cosine_similarity(emb1, emb2):
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return np.dot(emb1, emb2) / (norm1 * norm2)


class DecompositionAgent:
    def __init__(
        self,
        embedding_model="mxbai-embed-large",
        llm_model="gemma3",
        num_rewrites=3,
        ollama_client=None,
        similarity_fn=None,
    ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.num_rewrites = num_rewrites
        self.ollama_client = ollama_client or OllamaClient()
        self.similarity_fn = similarity_fn or cosine_similarity

    def embed_query(self, plain_query):
        embedding = self.ollama_client.embed(
            model=self.embedding_model, input_text=plain_query
        )
        return embedding["embeddings"][0]

    # 2. Feed to a llm, rewrite query into determiner form, llm gives K outcomes
    def rewrite_query(self, query_to_rewrite):
        messages = [
            {
                "role": "system",
                "content": REWRITE_QUERY_SYSTEM_PROMPT.format(
                    num_rewrites=self.num_rewrites
                ),
            },
            {"role": "user", "content": query_to_rewrite},
        ]
        response = self.ollama_client.chat(model=self.llm_model, message=messages)

        rewritten_messages = dict()
        if response.message.content:
            content = response.message.content
            parts = [line for line in content.split("\n") if line.strip()]

            for i in range(len(parts)):
                if parts[i][0].isdigit():
                    match = re.match(r"^\d+\.\s+(.*)", parts[i])
                    if match:
                        rewritten_messages[i] = match.group(1).strip()
        return rewritten_messages

    # 4. Perform cosine-similarity search and pick the best one
    def select_best_rewrite(self, original_embedding, rewrite_candidates):
        # Embed each candidate
        rewritten_queries_embeddings = dict()
        for rewritten_query in rewrite_candidates.values():
            rewritten_embedding = self.embed_query(rewritten_query)
            rewritten_queries_embeddings[rewritten_query] = rewritten_embedding

        best_sim_score = -float("inf")
        best_candidate = 0  # default 0

        for rewrite_query, rewrite_query_emb in rewritten_queries_embeddings.items():
            sim_score = self.similarity_fn(rewrite_query_emb, original_embedding)
            if sim_score > best_sim_score:
                best_sim_score = sim_score
                best_candidate = rewrite_query

        return best_candidate

    # 5. Perform sanity check on rewrite query, for all the queries that passed the sanity check pick the one with the best score

    # 6. feed the final query into a llm and split into three sub-tasks
    def decompose_query(self, rewritten_query):
        sub_task_message = [
            {
                "role": "system",
                "content": DECOMPOSE_QUERY_SYSTEM_PROMPT,
            },
            {"role": "user", "content": rewritten_query},
        ]
        response = self.ollama_client.chat(
            model=self.llm_model, message=sub_task_message
        )
        sub_tasks = []
        if response.message.content:
            content = response.message.content
            parts = [line for line in content.split("\n") if line.strip()]

            for part in parts:
                if part[0].isdigit():
                    match = re.match(r"^\d+\.\s+(.*)", part)
                    if match:
                        sub_tasks.append(match.group(1).strip())
        return sub_tasks

    def process(self, query):
        # 1. Take in user query, embed it
        query_embedding = self.embed_query(query)
        logger.info("User query detected, processing...")

        # 2. Rewrite the original query and generate K candidates
        rewritten_queries_dict = self.rewrite_query(query)
        logger.info("Rewriting query...")

        # 3. Select the best one based on cosine similarity score
        best_rewritten_query = self.select_best_rewrite(
            query_embedding, rewritten_queries_dict
        )
        logger.info("Choosing the best rewrite")

        # 4. Decompose into sub-tasks
        sub_tasks = self.decompose_query(best_rewritten_query)
        logger.info("Decomposing into sub-tasks...")

        return {
            "original_query": query,
            "rewritten_queries": rewritten_queries_dict,
            "best_rewritten_query": best_rewritten_query,
            "sub_tasks": sub_tasks,
        }


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    user_query = "How many papers has the department of earth science and engineering at Imperial College published last month, what are the research problems in those papers?"
    decompose_agent = DecompositionAgent()
    result = decompose_agent.process(user_query)
    print(result["sub_tasks"])
