# ========================
# This test is not used in the final product.
# ========================

import json
import logging
import os
from datetime import datetime
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas import evaluate, EvaluationDataset
from pipeline.orchestrator_baseline import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pipeline_execution(question):
    try:
        result = pipeline(question)

        if isinstance(result, str):
            return {"contexts": [], "answers": result}
        elif isinstance(result, dict):
            return result
        else:
            logger.warning(f"Unexpected result type: {type(result)}")
            return {"contexts": [], "answers": str(result)}

    except Exception as e:
        logger.error(
            f"Pipeline execution failed for question: {question[:50]}... Error: {e}"
        )
        return {"contexts": [], "answers": "Error occurred during processing"}


def validate_result(results):
    valid_results = []
    for i, result in enumerate(results):
        if all(
            key in result
            for key in ["user_input", "retrieved_contexts", "response", "reference"]
        ):
            if result["retrieved_contexts"] and result["response"]:
                valid_results.append(result)
            else:
                logger.warning(f"Skipping incomplete result at index {i}")
        else:
            logger.warning(f"Missing required keys in result at index {i}")
    logger.info(f"Valid results: {len(valid_results)}/{len(results)}")
    return valid_results


with open("../test_data/qa_set_2.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

print(f"Successfully load {len(qa_pairs)} questions")

results = []
logger.info(f"Processing {len(qa_pairs)} questions...")


for i, qa_pair in enumerate(qa_pairs):
    logger.info(f"Processing question {i+1}/{len(qa_pairs)}")

    question = qa_pair["Question"]
    answer = qa_pair["Answer"]

    result = pipeline_execution(question)

    if result is None:
        logger.warning(f"Pipeline returned None for question: {question[:50]}...")
        continue

    if "contexts" not in result or "answers" not in result:
        logger.warning(f"Incomplete result for question: {question[:50]}...")
        continue

    ragas_result = {
        "user_input": question,
        "retrieved_contexts": result.get("contexts", []),
        "response": result.get("answers", ""),
        "reference": answer,
    }

    results.append(ragas_result)

results = validate_result(results)

if results:
    output_folder = "test_results"

    # save intermediate results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(
        os.path.join(output_folder, f"ragas_test_results_baseline_{timestamp}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    evaluation_dataset = EvaluationDataset.from_list(results)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    logger.info("Starting Ragas evaluation...")
    ragas_results = evaluate(evaluation_dataset, metrics=metrics)

    print("=== RAGAS EVALUATION RESULTS ===")
    print(ragas_results)

    try:
        with open(
            os.path.join(output_folder, f"ragas_raw_baseline_{timestamp}.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(str(ragas_results))

    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}")

else:
    logger.error("No valid results to evaluate")
