import json
import logging
import os.path
from collections import defaultdict
from pipeline.orchestrator_agent_wise_feedback import pipeline
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas import evaluate, EvaluationDataset
from datetime import datetime

logger = logging.getLogger(__name__)

with open("test_data/test_set_3.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

logger.info(f"Successfully loaded {len(qa_pairs)} questions.")

level_scores = defaultdict(list)

metrics = [faithfulness, answer_relevancy, context_precision, context_recall]


def run_pipeline():
    results = []
    for i, qa_pair in enumerate(qa_pairs):
        question = qa_pair["Question"]
        answer = qa_pair["Answer"]
        try:
            logger.info(f"Processing question {i + 1}: {question[:50]}...")
            result = pipeline(question)

            if result is None:
                logger.warning(
                    f"Pipeline returned None for question: {question[:50]}..."
                )
                continue

            if not isinstance(result, dict):
                logger.error(
                    f"Unexpected pipeline result type for question {i + 1}: {result}"
                )
                continue

            if "contexts" not in result or "answers" not in result:
                logger.warning(f"Incomplete result for question: {question[:50]}...")
                logger.warning(
                    f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}"
                )
                continue

            ragas_result = {
                "user_input": question,
                "retrieved_contexts": result.get("contexts", []),
                "response": result.get("answers", ""),
                "reference": answer,
                "iteration_scores": result.get("iteration_scores", []),
            }

            results.append(ragas_result)

        except Exception as e:
            logger.error(f"Pipeline execution failed for question {i + 1}")
            logger.error(f"Question: {question[:50]}...")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception message: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

    return results


res = run_pipeline()

if res:
    evaluation_dataset = EvaluationDataset.from_list(res)

    logger.info("Starting Ragas evaluation...")
    ragas_results = evaluate(evaluation_dataset, metrics=metrics)

    print("=== RAGAS EVALUATION RESULTS ===")
    print(ragas_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_folder = "test_results/bloom/qwen"
    os.makedirs(output_folder, exist_ok=True)

    # overall scores
    output_path = os.path.join(output_folder, f"level_ragas_scores_awf_{timestamp}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(ragas_results))
    logger.info(f"Ragas scores saved to {output_path}")

    # per-question outputs
    qa_output_path = os.path.join(output_folder, f"qa_outputs_awf_{timestamp}.json")
    with open(qa_output_path, "w", encoding="utf-8") as f_json:
        json.dump(res, f_json, ensure_ascii=False, indent=2)
    logger.info(f"QA outputs saved to {qa_output_path}")
