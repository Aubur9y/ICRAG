from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate, EvaluationDataset
import pandas as pd

from pipeline.orchestrator import pipeline

# test_queries = [
#     "What are the main topics in the uploaded notebooks?",
#     "List all available documents"
# ]

qa_pairs = [
    {
        "user_input": "What is the forward problem?",
        "reference": "The forward problem can be thought of as: given x what is y?"
    },
    {
        "user_input": """
        How are over-determined and under-determined systems both addressed using optimisation techniques, and how do their respective solution strategies differ?
        """,
        "reference": """
        Over-determined systems, where there are more equations than unknowns (m>n), are typically addressed using the Least Squares method. This involves finding the solution that minimizes the sum of the squared differences between the observed and predicted values—effectively projecting the observation vector onto the column space of the matrix.
        Under-determined systems, where there are fewer equations than unknowns (m<n), require finding a solution that satisfies the equations but also minimizes the norm of the solution vector—this is called the Minimum Norm Solution. This choice imposes an additional optimisation criterion because the solution space is not unique.
        Both scenarios transform the inversion problem into an optimisation one: least squares minimizes residuals, while the minimum norm approach selects the "smallest" solution from an infinite set.
        """
    }
]

results = []

for qa_pair in qa_pairs:
    question = qa_pair['user_input']
    answer = qa_pair['reference']
    result = pipeline(question)
    ragas_result = {
        "user_input": question,
        "retrieved_contexts": result.get("contexts"),
        "response": result.get("answers"),
        "reference": answer
    }

    results.append(ragas_result)

evaluation_dataset = EvaluationDataset.from_list(results)

metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

ragas_results = evaluate(evaluation_dataset, metrics=metrics)
print(ragas_results)