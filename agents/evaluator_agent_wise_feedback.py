import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from utils.model_service import chat_with_model_thinking, chat_with_model

load_dotenv()


class ComponentEvaluator:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        self.COHERENCE_PROMPT = """
        You are an evaluator specialised in evaluating coherency.
        Your job is to evaluate if the given answer to a question is clear, well-structured and logically coherent.
        Please evaluate the following aspects of the answer:
        1. whether there are logical links between sentences.
        2. whether there are logical transitions between paragraphs.
        3. whether the overall structure is clear enough.
        4. whether the presentation of ideas if consistent with no self-contradictory.

        Please rate the answer on a scale of 1-10 based on the above criterias,
        where 1 is extremely incoherent, and 10 is very coherent.
        Please also provide detailed feedback on the reasons for your score and suggestions for improvement.

        Remember, your very first line MUST be "COHERENCE EVALUATOR, FINAL SCORE: X/10" to ensure proper score extraction.
        """

        self.RELEVANCE_PROMPT = """
        You are an evaluator specialised in evaluating relevance of answers.
        Your job is to evaluate if the given answer is relevant to the user query, and whether or not it has leveraged the contexts provided.
        Please evaluate the following aspects of the answer:
        1. whether the answer directly address the user's question.
        2. whether the answer use information from the relevant context.
        3. whether the answer contain any irrelevant information.
        4. whether any key pieces of information from the context are missing.

        Please rate the answer on a scale of 1-10 based on the above criterias,
        where 1 indicated completely irrelevant and 10 indicates highly relevant.
        Please also provide detailed feedback on the reasons for your score and suggestions for improvement.

        Remember, your very first line MUST be "RELEVANCE EVALUATOR, FINAL SCORE: X/10" to ensure proper score extraction.
        """

        self.ACCURACY_PROMPT = """
        You are an evaluator specialised in evaluating accuracy of answers.
        Your job is to evaluate if the information in a given answer is accurate and consistent with the provided context.
        1. whether the factual statements in the answer is completely accurate.
        2. whether the content of the answer is consistent with the information in the context.
        3. whether there are any incorrect inferences or overinterpretations.
        4. whether the context information is correctly cited or used.

        Please rate the answer on a scale of 1-10 based on the above criterias,
        where 1 indicated significantly inaccurate and 10 indicates fully accurate.
        Please also provide detailed feedback on the reasons for your score and suggestions for improvement.

        Remember, your very first line MUST be "ACCURACY EVALUATOR, FINAL SCORE: X/10" to ensure proper score extraction.
        """

    def evaluate_decomposition(self, original_query, sub_tasks, final_contexts):
        prompt = """
        You are an evaluator specialised in assessing the quality of query decomposition.
        Evaluation Criteria:
        1.Whether the subtasks fully cover all aspects of the original query
        2.Whether the granularity of the subtasks is appropriate — neither too broad nor too fragmented 
        3.Whether the subtasks have logical relationships and prioritization
        4.Whether the decomposition helps retrieve relevant information

        Please provide a score from 1 to 10 and give concrete suggestions for improvement.
        The first line must be: "DECOMPOSITION SCORE: X/10"
        """

        content = f"""
        Original query: {original_query}
        Sub tasks: {sub_tasks}
        Final retrieved contexts: {len(final_contexts) if final_contexts else 0} items

        Please evaluate the decomposition quality based on the above information.
        """
        return self.get_evaluation(prompt, content)

    def evaluate_retrieval(self, sub_task, retrieval_details, expected_info_type):
        prompt = """
        You are an evaluator specialised in assessing information retrieval quality.
        Evaluation Criteria:
        1. Whether tool selection is appropriate (vector search vs graph search vs web search vs direct answer)
        2. Whether retrieved information is relevant to the sub-task
        3. Whether retrieval depth is sufficient
        4. Whether important information was missed

        Please provide a score from 1 to 10 and give concrete suggestions for improvement.
        The first line must be: "RETRIEVAL SCORE: X/10"
        """

        content = f""""
        Sub-task: {sub_task}
        Retrieval status: {retrieval_details.get('status', 'unknown')}
        Tool used: {retrieval_details.get('tool_used', 'unknown')}
        Retrieval result: {str(retrieval_details.get('tool_output', 'No output'))}

        Please evaluate the retrieval quality based on the above information.
        """

        return self.get_evaluation(prompt, content)

    def evaluate_generation(self, query, contexts, response):
        coherence_eval = self.evaluate_coherence(query, contexts, response)
        relevance_eval = self.evaluate_relevance(query, contexts, response)
        accuracy_eval = self.evaluate_accuracy(query, contexts, response)

        weights = {"coherence": 0.3, "relevance": 0.4, "accuracy": 0.3}
        total_score = (
            coherence_eval["score"] * weights["coherence"]
            + relevance_eval["score"] * weights["relevance"]
            + accuracy_eval["score"] * weights["accuracy"]
        )

        combined_feedback = f"""
        COMPREHENSIVE GENERATION EVALUATION:

        Coherence Score: {coherence_eval['score']}/10
        {coherence_eval['feedback']}

        Relevance Score: {relevance_eval['score']}/10
        {relevance_eval['feedback']}

        Accuracy Score: {accuracy_eval['score']}/10
        {accuracy_eval['feedback']}

        Overall Weighted Score: {total_score:.2f}/10
        """

        return {
            "score": total_score,
            "feedback": combined_feedback,
            "component_scores": {
                "coherence": coherence_eval["score"],
                "relevance": relevance_eval["score"],
                "accuracy": accuracy_eval["score"],
            },
        }

    def evaluate_coherence(self, query, contexts, response):
        return self.get_evaluation(
            self.COHERENCE_PROMPT,
            f"""
            User query: {query}
            Provided: contexts: {contexts}
            Generated response: {response}
            Please evaluate based on the above information.
            """,
            score_pattern=r"FINAL SCORE:\s*(\d+(?:\.\d+)?)/10",
        )

    def evaluate_relevance(self, query, contexts, response):
        return self.get_evaluation(
            self.RELEVANCE_PROMPT,
            f"""
            User query: {query}
            Provided: contexts: {contexts}
            Generated response: {response}
            Please evaluate based on the above information.
            """,
            score_pattern=r"FINAL SCORE:\s*(\d+(?:\.\d+)?)/10",
        )

    def evaluate_accuracy(self, query, contexts, response):
        return self.get_evaluation(
            self.ACCURACY_PROMPT,
            f"""
            User query: {query}
            Provided: contexts: {contexts}
            Generated response: {response}
            Please evaluate based on the above information.
            """,
            score_pattern=r"FINAL SCORE:\s*(\d+(?:\.\d+)?)/10",
        )

    def get_evaluation(self, system_prompt, user_context, score_pattern=None):
        # messages = [
        #     ChatCompletionSystemMessageParam(content=system_prompt, role="system"),
        #     ChatCompletionUserMessageParam(content=user_context, role="user"),
        # ]
        #
        # result = self.client.chat.completions.create(
        #     model="gpt-3.5-turbo", messages=messages, temperature=0.3
        # )
        # content = result.choices[0].message.content

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_context},
        ]

        content = chat_with_model(messages=messages)

        if score_pattern:
            score_match = re.search(score_pattern, content)
        else:
            score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)/10", content)

        if score_match:
            score = float(score_match.group(1))
        else:
            score_match = re.search(
                r"score[:\s]+(\d+(?:\.\d+)?)", content, re.IGNORECASE
            )
            if score_match:
                score = float(score_match.group(1))
            else:
                number_matches = re.findall(r"\b([1-9]|10)(?:\.\d+)?\b", content)
                score = float(number_matches[0]) if number_matches else 5.0

        return {"score": score, "feedback": content}

    def evaluate_pipeline(
        self,
        original_query,
        sub_tasks,
        retrieval_results,
        final_contexts,
        final_response,
    ):
        evaluations = {}

        evaluations["decomposition"] = self.evaluate_decomposition(
            original_query, sub_tasks, final_contexts
        )

        evaluations["retrieval"] = {}
        for i, (sub_task, retrieval_details) in enumerate(
            zip(sub_tasks, retrieval_results)
        ):
            evaluations["retrieval"][f"subtask_{i}"] = self.evaluate_retrieval(
                sub_task, retrieval_details, "unknown"
            )

        evaluations["generation"] = self.evaluate_generation(
            original_query, final_contexts, final_response
        )

        decomp_score = evaluations["decomposition"]["score"]
        avg_retrieval_score = sum(
            eval_data["score"] for eval_data in evaluations["retrieval"].values()
        ) / len(evaluations["retrieval"])
        generation_score = evaluations["generation"]["score"]

        weights = {"decomposition": 0.2, "retrieval": 0.3, "generation": 0.5}
        total_score = (
            decomp_score * weights["decomposition"]
            + avg_retrieval_score * weights["retrieval"]
            + generation_score * weights["generation"]
        )

        evaluations["overall"] = {
            "score": total_score,
            "component_breakdown": {
                "decomposition": decomp_score,
                "average_retrieval": avg_retrieval_score,
                "generation": generation_score,
            },
        }

        return evaluations
