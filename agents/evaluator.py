import os
import re

from dotenv import load_dotenv
from langchain_experimental.graph_transformers.llm import system_prompt
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from sympy.physics.units import temperature

load_dotenv()

def create_evaluation_agent(type):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    COHERENCE_PROMPT="""
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
    
    Remember, your very first line MUST be "FINAL SCORE: X/10" to ensure proper score extraction.
    """

    RELEVANCE_PROMPT="""
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
    
    Remember, your very first line MUST be "FINAL SCORE: X/10" to ensure proper score extraction.
    """

    ACCURACY_PROMPT="""
    You are an evaluator specialised in evaluating accuracy of answers.
    Your job is to evaluate if the information in a given answer is accurate and consistent with the provided context.
    1. whether the factual statements in the answer is completely accurate.
    2. whether the content of the answer is consistent with the information in the context.
    3. whether there are any incorrect inferences or overinterpretations.
    4. whether the context information is correctly cited or used.
    
    Please rate the answer on a scale of 1-10 based on the above criterias, 
    where 1 indicated significantly inaccurate and 10 indicates fully accurate.
    Please also provide detailed feedback on the reasons for your score and suggestions for improvement.
    
    Remember, your very first line MUST be "FINAL SCORE: X/10" to ensure proper score extraction.
    """

    if type == "coherence":
        system_prompt = COHERENCE_PROMPT
    elif type == "relevance":
        system_prompt = RELEVANCE_PROMPT
    elif type == "accurate":
        system_prompt = ACCURACY_PROMPT

    def evaluate(query, contexts, response):
        messages = [
            ChatCompletionSystemMessageParam(content=system_prompt, role="system"),
            ChatCompletionUserMessageParam(content=f"""
            User query: {query}
            
            Provided contexts:
            {contexts}
            
            Generated response:
            {response}
            
            Please evaluate based on the above information.
            """, role="user")
        ]

        result = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3
        )

        content = result.choices[0].message.content

        score_match = re.search(r'FINAL SCORE:\s*(\d+(?:\.\d+)?)/10', content)
        if score_match:
            score = float(score_match.group(1))
        else:
            score_match = re.search(r'score[:\s]+(\d+(?:\.\d+)?)', content, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
            else:
                number_matches = re.findall(r'\b([1-9]|10)(?:\.\d+)?\b', content)
                score = float(number_matches[0]) if number_matches else 5.0
        return {
            "score": score,
            "feedback": content
        }
    return evaluate


class Evaluator:
    def __init__(self, query, context, response):
        self.query = query
        self.context = context
        self.response = response
        self.evaluator_agents = [
            create_evaluation_agent("coherence"),
            create_evaluation_agent("relevance"),
            create_evaluation_agent("accurate")
        ]

    def evaluate(self):
        scores = []
        feedback = []

        for agent in self.evaluator_agents:
            result = agent(self.query, self.context, self.response)
            scores.append(result["score"])
            feedback.append(result["feedback"])
        avg_score = sum(scores) / len(scores)
        return {
            "score": avg_score,
            "feedback": feedback,
            "needs_improvement": avg_score < 7.0
        }


def test_evaluator():
    # Sample user query
    query = "What are the main advantages and limitations of transformer models in NLP?"

    # Context information that would be retrieved
    context = """
    Transformer models were introduced in the paper "Attention Is All You Need" by Vaswani et al. in 2017. They have become the foundation of modern natural language processing (NLP) systems. 

    The main advantages of transformer models include:
    1. Parallelization: Unlike RNNs, transformers process all tokens simultaneously, allowing for much faster training on modern GPUs.
    2. Long-range dependencies: The self-attention mechanism allows transformers to capture relationships between words regardless of their distance in the text.
    3. Transfer learning: Pre-trained transformer models like BERT, GPT, and T5 can be fine-tuned for specific tasks with relatively small amounts of task-specific data.
    4. Scalability: Transformer models have shown impressive scaling properties, with performance continuing to improve as model size increases (as seen with models like GPT-3, GPT-4, etc.).

    However, transformers have several limitations:
    1. Computational complexity: The self-attention mechanism has quadratic complexity with respect to sequence length, making it expensive to process very long documents.
    2. Context window limitations: Most transformer models have a fixed context window (e.g., 512, 1024, or 2048 tokens), limiting their ability to process long texts.
    3. Position encoding limitations: The original position encoding scheme isn't well-suited for tasks requiring precise positional understanding.
    4. Data hunger: Large transformer models require massive amounts of training data to perform well.
    5. Interpretability challenges: The attention mechanisms, while providing some insight, don't always offer clear explanations of model decisions.

    Recent innovations like Sparse Transformers, Longformer, and Reformer have addressed some of these limitations, particularly around handling longer sequences more efficiently.
    """

    # Sample response to be evaluated
    response = """
    Transformer models have several advantages in NLP. They can process text in parallel instead of sequentially like RNNs, which makes them faster to train. The self-attention mechanism helps capture relationships between words that are far apart in a text.

    Another advantage is that transformer models can be pre-trained on large amounts of text and then fine-tuned for specific tasks. This approach has led to models like BERT and GPT which achieve state-of-the-art results on many NLP benchmarks.

    The main limitation of transformers is that they struggle with very long texts because the complexity of self-attention grows quadratically with sequence length. They also require large amounts of training data and computational resources.

    Additionally, transformers have difficulty with tasks that require precise understanding of hierarchical structures in language. The models can sometimes generate fluent but factually incorrect or nonsensical text.

    Overall, despite their limitations, transformer models have revolutionized NLP and continue to be the foundation for most advanced language technologies today.
    """

    # Create and run the evaluator
    evaluator = Evaluator(query, context, response)
    evaluation_result = evaluator.evaluate()

    # Print the results
    print(f"Overall Score: {evaluation_result['score']}")
    print(f"Needs Improvement: {evaluation_result['needs_improvement']}")

    print("\n--- Feedback ---")
    for i, feedback in enumerate(evaluation_result['feedback']):
        print(f"\nEvaluator {i + 1}:\n{feedback}")

    return evaluation_result


if __name__ == "__main__":
    test_evaluator()