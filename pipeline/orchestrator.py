from agents.brain_agent import BrainAgent
from agents.decision_agent import DecisionAgent
from agents.decomposition_agent import DecompositionAgent
from agents.evaluator import Evaluator
import logging
import re
import tiktoken


class ToolReturnFilter(logging.Filter):
    def __init__(self, name=''):
        super().__init__(name)
        self.pattern = re.compile(r"^Tool '.*' returns:")

    def filter(self, record):
        return not self.pattern.match(record.getMessage())

logging.basicConfig(
    level=logging.INFO
)
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
        return encoding.decode(truncated_tokens, errors='ignore')
    return text


def pipeline(user_query, max_retries=2):
    collections = "ipynb_embeddings" # TODO: change this later

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

    pipeline_results = []

    # using brain agent on each sub-task
    for i, sub_task_query in enumerate(sub_tasks):
        logger.info(f"Step 2.{i+1}: Processing sub-task ({i+1}/{len(sub_tasks)}): \"{sub_task_query}\"")

        try:
            retrieval_info = brain_agent.retrieve(sub_task_query)
            pipeline_results.append({
                "sub_task_query": sub_task_query,
                "retrieval_details": retrieval_info
            })
            logger.info(f"Sub-task {i+1} finished. Status: {retrieval_info.get('status')}")

        except Exception as e:
            logger.error(f"Error processing sub-task \"{sub_task_query}\": {e}", exc_info=True)
            pipeline_results.append({
                "sub_task_query": sub_task_query,
                "retrieval_details": {
                    "status": "pipeline_error",
                    "message": str(e),
                }
            })

    # prepare context for generation
    logger.info("Step 3: Aggregating context from sub-tasks.")
    contexts_for_generation = []
    for result_item in pipeline_results:
        details = result_item.get('retrieval_details', {})
        if details.get('status') == 'no_relevant_information':
            logger.warning(f"No relevant information found for sub-task: {result_item.get('sub_task_query')}")
        if details.get('status') == 'tool_executed':
            tool_output = details.get('tool_output')
            if tool_output:
                contexts_for_generation.append(str(tool_output))
        elif details.get('status') == 'direct_answer':
            direct_answer = details.get('direct_answer_content')
            if direct_answer:
                contexts_for_generation.append(str(direct_answer))

    has_relevant_info = any(
        result_item.get('retrieval_details', {}).get('status') not in ['no_relevant_information', 'error']
        for result_item in pipeline_results
    )

    if not has_relevant_info:
        return "Sorry, I couldn't find any information related to your query in my knowledge base."

    final_context_str = "\n\n---\n\n".join(contexts_for_generation)

    # make decision
    decision_agent = DecisionAgent()
    current_response_content = ""
    last_feedback = None

    for attempt in range(max_retries + 1):
        logger.info(f"Step 4.{attempt + 1}: Generation attempt {attempt + 1}/{max_retries + 1}")

        if attempt == 0:
            prompt_for_generation = f"""
            Answer the question: "{best_rewritten_query}" based on the following context:
            context: {final_context_str}
            
            Please provide an accurate and comprehensive response based on the above contexts.
            """
        else:
            # calibration prompt
            logger.info("Constructing calibration prompt with feedback.")
            prompt_for_generation = f"""
            Your previous attempt to answer the question "{best_rewritten_query}" was not sufficient.
            
            Here is the feedback on your last answer:
            --- FEEDBACK ---
            {last_feedback}
            --- END FEEDBACK ---
            
            Here is the original context again:
            --- CONTEXT ---
            {final_context_str}
            --- END CONTEXT ---
            
            Please generate a new, improved answer that directly addresses the feedback and better utilises the provided context.
            """

        generated_response = decision_agent.generate(prompt_for_generation)
        current_response_content = generated_response.content

        logger.info(f"Step 5.{attempt + 1}: Evaluating the generated response.")

        EVALUATOR_MODEL = "gpt-3.5-turbo"
        MAX_CONTEXT_TOKENS_FOR_EVALUATOR = 12000

        truncated_context_for_eval = truncate_text_by_tokens(
            final_context_str,
            EVALUATOR_MODEL,
            MAX_CONTEXT_TOKENS_FOR_EVALUATOR
        )

        evaluator = Evaluator(best_rewritten_query, truncated_context_for_eval, current_response_content)
        evaluation_result = evaluator.evaluate()

        logger.info(f"Attempt {attempt + 1} evaluation score: {evaluation_result['score']:.2f}")

        if not evaluation_result['needs_improvement']:
            logger.info("Generated response meets quality standards.")
            break
        else:
            last_feedback = "\n".join(evaluation_result['feedback'])
            if attempt < max_retries:
                logger.info(f"Response quality is low. Retrying... ({attempt + 1}/{max_retries})")
            else:
                logger.info("Max retries reached. Returning the last generated response.")

    logger.info("Pipeline finished.")
    return current_response_content

if __name__ == "__main__":
    query = "Can you summarize the tasks needed to do the ads project regarding the flood risk, in the steps needed, also what is the technical requirements for each task?"

    results = pipeline(query)

    print("\n" + "=" * 25 + " Final Generated Answer " + "=" * 25 + "\n")
    print(results)
    print("\n" + "=" * 72)