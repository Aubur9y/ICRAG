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
from pipeline.orchestrator_agent_wise_feedback import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pipeline_execution(question):
    try:
        result = pipeline(question)

        # 如果返回的是字符串，需要构造合适的格式
        if isinstance(result, str):
            return {"contexts": [], "answers": result, "question": question}

        if isinstance(result, dict):
            if "contexts" in result and "answers" in result:
                return result
            else:
                logger.warning(
                    f"Incomplete result structure for question: {question[:50]}..."
                )
                return None

        logger.warning(f"Unexpected result type: {type(result)}")
        return None

    except Exception as e:
        logger.error(
            f"Pipeline execution failed for question: {question[:50]}... Error: {e}"
        )
        return None


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


# =====
# Prompt for generating the qa pairs (gpt4o):
# Your task is to formulate a set of questions from the given documents.
#
# The question set should contain 35 percent of simple question, 45 percent of medium difficulty questions and 20 percent of hard question.
#
# Simple question is the question that can be answered when the right word or phrase is found. Medium question is the question that requires the correct retrieval of two or three contexts to be answer accurately, medium question are always within the same document. Hard question is the question that need three or more contexts retrieved correctly in order to answer accurately, or it can be the question that requires two context but are from different document.
#
# You should also decide how many questions to formulate based on the size and length of the provided documents.
# qa_pairs = [
#     {
#         "question": "What is the primary objective of using climate models?",
#         "reference": "To quantify, forecast, and understand the impacts, adaptation strategies, and mitigation options related to climate change.",
#     },
#     {
#         "question": "What does MIP stand for in climate modeling?",
#         "reference": "Model Intercomparison Project",
#     },
#     {
#         "question": "What do the variables T, precipitation, and sea level represent in climate risk analysis?",
#         "reference": "They represent different climate variables that define the magnitude, frequency, duration, and spatial extent of climate extremes and risks.",
#     },
#     {
#         "question": "What is the function of discretization in numerical modeling?",
#         "reference": "Discretization converts continuous equations into discrete forms that can be solved numerically using computers.",
#     },
#     {
#         "question": "What is the difference between forward and backward Euler methods?",
#         "reference": "Forward Euler is an explicit method using current values, while Backward Euler is an implicit method requiring solving an equation with future values.",
#     },
#     {
#         "question": "What are the main exchange properties at the interfaces of coupled climate models?",
#         "reference": "Heat, water, momentum, and energy.",
#     },
#     {
#         "question": "What is a “coupler” in a climate model?",
#         "reference": "The central component that facilitates the exchange of properties between different modules such as atmosphere, ocean, and land.",
#     },
#     {
#         "question": "What does increasing model resolution imply for temporal resolution?",
#         "reference": "It requires decreasing the time step to maintain numerical stability.",
#     },
#     {
#         "question": "What is relative humidity defined as in climate science?",
#         "reference": "The ratio of partial pressure of water vapor to the equilibrium partial pressure at a given temperature.",
#     },
#     {
#         "question": "Why is cloud modeling challenging in climate simulations?",
#         "reference": "Because cloud microphysics are poorly understood and occur on spatial and temporal scales much smaller than model grid cells.",
#     },
#     {
#         "question": "What are reanalysis products in the context of climate models?",
#         "reference": "Climate model outputs fitted to past observational data using data assimilation techniques.",
#     },
#     {
#         "question": "What does the acronym OSSE stand for?",
#         "reference": "Observational System Simulation Experiments",
#     },
#     {
#         "question": "What is meant by “internal variability” in ensemble simulations?",
#         "reference": "The natural fluctuations in climate that occur without changes in external forcing.",
#     },
#     {
#         "question": "Who is Syukuro Manabe, and what is his contribution to climate modeling?",
#         "reference": "A Nobel laureate known for pioneering work in simplifying climate models to capture essential processes.",
#     },
#     {
#         "question": "What are the typical learning objectives of an environmental data module focused on climate science?",
#         "reference": "Understanding data formats, repositories, statistical tools, model limitations, and strategies for interpreting uncertainties.",
#     },
#     {
#         "question": "How can climate models be used for attribution of extreme weather events?",
#         "reference": "By comparing model simulations with and without human influence to determine the likelihood of observed events.",
#     },
#     {
#         "question": "How is discretization applied to model time and space in numerical simulations?",
#         "reference": "By defining step sizes (Δx, Δy, Δz, Δt) and applying finite-difference schemes to solve partial differential equations.",
#     },
#     {
#         "question": "What are the key differences between forward, backward, and centered difference methods in numerical modeling?",
#         "reference": "They differ in how derivatives are approximated: forward uses current and next step, backward uses current and previous, and centered uses both surrounding points.",
#     },
#     {
#         "question": "Why is numerical diffusion a concern in modeling, and how can idealized simulations reveal it?",
#         "reference": "Numerical diffusion causes artificial smoothing of variables; idealized simulations can show deviations from expected behavior when models are reversed in time.",
#     },
#     {
#         "question": "What is meant by filtering equations during discretization and how does it impact model complexity?",
#         "reference": "It involves omitting small-scale physics irrelevant to the model's spatial/temporal resolution, simplifying computation.",
#     },
#     {
#         "question": "How do coupled climate models handle differences in module grid structures and time-scales?",
#         "reference": "Through the coupler, which interpolates and conserves fluxes while synchronizing modules with differing spatial grids and time-steps.",
#     },
#     {
#         "question": "What are the implications of nonlinear feedbacks and teleconnections for climate model design?",
#         "reference": "They necessitate coupling because distant regions and processes influence each other non-linearly, impacting long-term predictions.",
#     },
#     {
#         "question": "Why is substantial data management infrastructure necessary for climate modeling?",
#         "reference": "Due to the volume and complexity of inputs, outputs, and interpreted products generated by models.",
#     },
#     {
#         "question": "What are the costs associated with increasing model resolution in climate simulations?",
#         "reference": "It multiplies computational time, memory use, and energy requirements exponentially.",
#     },
#     {
#         "question": "How does increasing spatial resolution affect the representation of topography and gradients?",
#         "reference": "It allows more accurate modeling of features like mountains and enhances pressure gradient calculations.",
#     },
#     {
#         "question": "How do subgrid-scale processes affect the accuracy and usefulness of climate models?",
#         "reference": "They must be parameterized or resolved to capture critical dynamics like eddies and convection, impacting model realism.",
#     },
#     {
#         "question": "What are the advantages and limitations of using reanalysis products in climate modeling?",
#         "reference": "They provide high-resolution past climate data but are still model outputs limited by the quality of the underlying model and data availability.",
#     },
#     {
#         "question": "Why is model comparison with observational data non-trivial?",
#         "reference": "Because model grid-averaged outputs may not align with point-based observations in time or space.",
#     },
#     {
#         "question": "What are some typical misconceptions about the use of climate models?",
#         "reference": "That models must exactly match data to be useful or that models without direct measurements are invalid.",
#     },
#     {
#         "question": "How does the resolution of climate models affect ensemble simulation strategies?",
#         "reference": "Higher resolution limits the number of ensemble runs due to computational cost, while lower resolution allows more ensemble simulations.",
#     },
#     {
#         "question": "What considerations are involved in selecting a climate model for a specific analysis purpose?",
#         "reference": "Model complexity, resolution, data availability, and the nature of the climate process being studied.",
#     },
#     {
#         "question": "What does the concept “all models are wrong, but some are useful” imply in climate modeling?",
#         "reference": "That while no model is perfect, they can still provide valuable insights and guide decision-making if used properly.",
#     },
#     {
#         "question": "How do the differences in numerical integration schemes affect climate model outputs and how can idealized simulations be used to test them?",
#         "reference": "Different schemes (forward, backward, centered) produce divergent outcomes due to varying numerical diffusion; idealized reversibility tests help identify such artifacts.",
#     },
#     {
#         "question": "How do climate models help in designing adaptation and mitigation policies, and how do uncertainties impact their utility?",
#         "reference": "They simulate future scenarios to inform policy, but model limitations and input uncertainties mean outcomes are probabilistic, not deterministic.",
#     },
#     {
#         "question": "Explain the challenge of aligning atmospheric and oceanic grids in coupled climate models, and how discretization plays a role.",
#         "reference": "Grid mismatch leads to flux interpolation errors; discretization choices determine resolution and time-step compatibility, influencing accuracy.",
#     },
#     {
#         "question": "How do ensemble simulations help distinguish between internal variability and forced climate responses?",
#         "reference": "By running multiple scenarios with identical forcings, ensemble simulations reveal natural variability, helping isolate external influences.",
#     },
#     {
#         "question": "Discuss the trade-offs between model complexity and resolution, and their scientific and computational implications.",
#         "reference": "Higher complexity improves realism but raises computational cost; simplified models are faster but may miss key processes.",
#     },
#     {
#         "question": "How do model parameterizations influence forecasting accuracy, and why is experimental design essential?",
#         "reference": "Parameterizations simplify unresolved processes, but incorrect assumptions lead to biases; careful experimental design tests robustness.",
#     },
#     {
#         "question": "How should observational data be compared to model outputs, considering concepts like OSSE and reanalysis?",
#         "reference": "Observations must be matched in resolution and timing; OSSE and reanalysis integrate data with models but still carry model-based limitations.",
#     },
#     {
#         "question": "What are the difficulties of integrating multiple variables (e.g., temperature, precipitation) across model modules, and how does resolution affect this integration?",
#         "reference": "Different modules use distinct grids and time-scales, making consistent integration of multiple variables complex; resolution mismatches exacerbate this.",
#     },
# ]

with open("test_data/qa_set_1.json", "r", encoding="utf-8") as f:
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
        os.path.join(output_folder, f"ragas_test_results_awf_{timestamp}.json"),
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

    metric_scores = ragas_results.scores

    try:
        # results_dict = {
        #     "metrics": metric_scores,
        #     "timestamp": timestamp,
        #     "total_questions": len(qa_pairs),
        #     "valid_results": len(results),
        # }
        #
        # logger.info(
        #     f"Saving evaluation results to ragas_agent_wise_feedback_{timestamp}.json"
        # )
        # logger.info(f"Results dict: {results_dict}")
        #
        # # save evaluation results
        # with open(
        #     os.path.join(output_folder, f"ragas_agent_wise_feedback_{timestamp}.json"),
        #     "w",
        #     encoding="utf-8",
        # ) as f:
        #     json.dump(results_dict, f, ensure_ascii=False, indent=2)
        #
        # logger.info(
        #     f"Evaluation results saved to ragas_agent_wise_feedback_{timestamp}.json"
        # )

        with open(
            os.path.join(
                output_folder, f"ragas_agent_wise_feedback_raw_{timestamp}.txt"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(str(ragas_results))

    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}")
        # # backup saving
        # with open(
        #     os.path.join(
        #         output_folder, f"ragas_agent_wise_feedback_raw_{timestamp}.txt"
        #     ),
        #     "w",
        #     encoding="utf-8",
        # ) as f:
        #     f.write(str(ragas_results))
else:
    logger.error("No valid results to evaluate")
