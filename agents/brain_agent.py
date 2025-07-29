import os
import json
import logging
import re
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionMessage,
)
from dotenv import load_dotenv
from sympy.physics.units import force

from retrievers.graph_retriever import GraphRAGProcessor
from retrievers.vector_retriever import VectorRetrieverAgent
from retrievers.web_retriever import WebRetrieverAgent
from utils.prompt_templates import BRAIN_QUERY_TOOL_USE_SYSTEM_PROMPT

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# brain agent is to choose the best retrieval strategy and adjust on feedback, it's fundamentally different from
# orchestrator which is the whole pipeline
class BrainAgent:
    MODEL_NAME = "gpt-4o"
    VECTOR_DB_COLLECTION = [
        "ipynb_embeddings",
        "code_embeddings",
        "pdf_embeddings",
        "text_embeddings",
    ]

    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.tool_specs = [
            {
                "type": "function",
                "function": {
                    "name": "vector_search",
                    "description": "Use this for knowledge-base or conceptual queries",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "graph_search",
                    "description": "Use this for entity relationship queries and structured knowledge retrieval",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search",
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of relevant nodes to return",
                                "default": 10,
                            },
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Use this for real-time or current events",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
        ]
        self.ToolDecidePrompt = BRAIN_QUERY_TOOL_USE_SYSTEM_PROMPT
        self.vector_retriever = VectorRetrieverAgent(
            collections=self.VECTOR_DB_COLLECTION
        )
        self.graph_retriever = GraphRAGProcessor()
        self.web_retriever = WebRetrieverAgent()

    def decide_tool(self, query) -> ChatCompletionMessage:
        messages = [
            ChatCompletionSystemMessageParam(
                role="system", content=self.ToolDecidePrompt
            ),
            ChatCompletionUserMessageParam(role="user", content=query),
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=self.tool_specs,
            tool_choice="auto",
        )
        """
        openai response format is like this: 
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "vector_search", (or web_search)
                        "arguments": '{"query": "the query"}'
                    }
                }
            ]
        }

        or for graph search
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "graph_search",
                        "arguments": '{
                            "query": "the query"
                            "k": whatever number it decided (default is 10)
                        }'
                    }
                }
            ]
        }
        """

        return response.choices[0].message

    def execute_tool(self, tool_name, tool_args_dict):
        if isinstance(tool_args_dict, dict) and "query" in tool_args_dict:
            tool_query = tool_args_dict["query"]
        elif isinstance(tool_args_dict, str):
            tool_query = tool_args_dict
        else:
            tool_query = str(tool_args_dict)

        logger.info(f"Executing tool: {tool_name}")
        if tool_name == "vector_search":
            result = self.vector_retriever.retrieve(tool_query)
        elif tool_name == "graph_search":
            k = tool_args_dict.get("k", 10)
            result = self.graph_retriever.query_graph(tool_query, k=k)
        elif tool_name == "web_search":
            search_result = self.web_retriever.retrieve(tool_query).get("results", [])
            if not search_result:
                result = "No web search results found."
            else:
                content_parts = []
                for i, res in enumerate(search_result[:3]):
                    content_parts.append(
                        f"Source {i + 1}: {res.get('content', 'No content available')}"
                    )
                result = "\n\n".join(content_parts)
        else:
            result = "Unknown tool call."

        return result

    # retrieve contexts for each query
    def retrieve(self, query):
        # decide which tools to use
        assistant_initial_message = self.decide_tool(query)

        # build the context
        context_messages = [
            ChatCompletionSystemMessageParam(
                role="system", content=self.ToolDecidePrompt
            ),
            ChatCompletionUserMessageParam(role="user", content=query),
            assistant_initial_message,
        ]

        if (
            hasattr(assistant_initial_message, "tool_calls")
            and assistant_initial_message.tool_calls
        ):
            tool_call = assistant_initial_message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments

            try:
                tool_args_dict = json.loads(
                    tool_args_str
                )  # tool args is the task the tool is to solve
            except json.JSONDecodeError:
                logger.error(f"Error: Decoding tool argument failed: {tool_args_str}")
                return {
                    "status": "error",
                    "message": "Decoding tool argument failed.",
                    "context_messages_for_generation": context_messages,
                }

            logger.info(
                f"Tool decision: use tool '{tool_name}', arguments: {tool_args_str}"
            )

            tool_output_content = self.execute_tool(tool_name, tool_args_dict)

            if (
                isinstance(tool_output_content, dict)
                and tool_output_content.get("status") == "no_relevant_information_found"
            ):
                return {
                    "status": "no_relevant_information",
                    "message": "No relevant information found in knowledge base.",
                    "context_messages_for_generation": context_messages,
                }

            tool_result_message = ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=tool_call.id,
                content=(
                    json.dumps(tool_output_content)
                    if not isinstance(tool_output_content, str)
                    else tool_output_content
                ),
            )
            context_messages.append(tool_result_message)

            return {
                "status": "tool_executed",
                "tool_name": tool_name,
                "tool_args": tool_args_dict,
                "tool_output": tool_output_content,  # this is the actual context used for generation
                "context_messages_for_generation": context_messages,  # this is the full context of the model
            }
        # if the model decides no tool is needed, it will generate a response stating it
        else:
            logger.info("No tools used.")
            return {
                "status": "direct_answer",
                "direct_answer_content": assistant_initial_message.content,
                "context_messages_for_generation": context_messages,
            }

    def analyse_feedback_for_tool_selection(self, feedback):
        feedback_lower = feedback.lower()
        if "wrong tool" in feedback_lower or "tool selection" in feedback_lower:
            if "vector search" in feedback_lower and "should use" in feedback_lower:
                return "vector_search"
            elif "graph search" in feedback_lower and "should use" in feedback_lower:
                return "graph_search"
            elif "web search" in feedback_lower and "should use" in feedback_lower:
                return "web_search"
            elif "direct answer" in feedback_lower and "should use" in feedback_lower:
                return "direct_answer"

        if (
            "current information" in feedback_lower
            or "real-time" in feedback_lower
            or "recent" in feedback_lower
        ):
            return "web_search"
        elif "not relevant" in feedback_lower or "missed information" in feedback_lower:
            return "broader_search"
        elif "too shallow" in feedback_lower or "insufficient depth" in feedback_lower:
            return "deeper_search"
        elif "outdated" in feedback_lower or "old information" in feedback_lower:
            return "web_search"

        return None

    def retrieve_with_feedback(self, query, feedback):
        logger.info(f"Processing query with retrieval feedback: {query}")

        tool_preference = self.analyse_feedback_for_tool_selection(feedback)

        if tool_preference:
            logger.info(f"Feedback suggests using: {tool_preference}")

            if tool_preference in ["vector_search", "graph_search", "web_search"]:
                return self.execute_tool_with_feedback(query, tool_preference, feedback)
            elif tool_preference == "broader_search":
                result = self.execute_tool_with_feedback(
                    query, "graph_search", feedback
                )
                if result.get("status") == "no_relevant_information":
                    return self.execute_tool_with_feedback(
                        query, "vector_search", feedback
                    )
                return result
            elif tool_preference == "deeper_search":
                enhanced_query = self.enhance_search_query(query, feedback)
                return self.execute_tool_with_feedback(
                    enhanced_query, "vector_search", feedback
                )
            elif tool_preference == "direct_answer":
                return self.generate_direct_answer_with_feedback(query, feedback)

        return self.retrieve_with_enhanced_decision(query, feedback)

    def execute_tool_with_feedback(self, query, forced_tool, feedback):
        enhanced_query = self.enhance_search_query(query, feedback)

        if forced_tool == "vector_search":
            tool_args = {"query": enhanced_query}
            tool_output = self.execute_tool("vector_search", tool_args)
        elif forced_tool == "graph_search":
            k = 15 if "more comprehensive" in feedback.lower() else 10
            tool_args = {"query": enhanced_query, "k": k}
            tool_output = self.execute_tool("graph_search", tool_args)
        elif forced_tool == "web_search":
            tool_args = {"query": enhanced_query}
            tool_output = self.execute_tool("web_search", tool_args)
        else:
            return {"status": "error", "message": f"Unknown forced tool: {forced_tool}"}

        return {
            "status": "tool_executed",
            "tool_name": forced_tool,
            "tool_args": tool_args,
            "tool_output": tool_output,
            "enhanced_query": enhanced_query,
            "feedback_applied": True,
        }

    def retrieve_with_enhanced_decision(self, query, feedback):
        enhanced_prompt = self.enhance_tool_decision_prompt(feedback)

        messages = [
            ChatCompletionSystemMessageParam(role="system", content=enhanced_prompt),
            ChatCompletionUserMessageParam(role="user", content=query),
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=self.tool_specs,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message
        context_messages = [
            ChatCompletionSystemMessageParam(role="system", content=enhanced_prompt),
            ChatCompletionUserMessageParam(role="user", content=query),
            assistant_message,
        ]

        if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
            tool_call = assistant_message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments

            try:
                tool_args_dict = json.loads(tool_args_str)
            except json.JSONDecodeError:
                logger.error(f"Error: Decoding tool argument failed: {tool_args_str}")
                return {
                    "status": "error",
                    "message": "Decoding tool argument failed.",
                    "feedback_applied": True,
                }

            if "query" in tool_args_dict:
                tool_args_dict["query"] = self.enhance_search_query(
                    tool_args_dict["query"], feedback
                )

            tool_output_content = self.execute_tool(tool_name, tool_args_dict)

            tool_result_message = ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=tool_call.id,
                content=(
                    json.dumps(tool_output_content)
                    if not isinstance(tool_output_content, str)
                    else tool_output_content
                ),
            )
            context_messages.append(tool_result_message)

            return {
                "status": "tool_executed",
                "tool_name": tool_name,
                "tool_args": tool_args_dict,
                "tool_output": tool_output_content,
                "context_messages_for_generation": context_messages,
                "feedback_applied": True,
            }
        else:
            return {
                "status": "direct_answer",
                "direct_answer_content": assistant_message.content,
                "context_messages_for_generation": context_messages,
                "feedback_applied": True,
            }

    def enhance_tool_decision_prompt(self, feedback):
        base_prompt = self.ToolDecidePrompt

        if "web search" in feedback.lower() and "should use" in feedback.lower():
            base_prompt += "\nPRIORITY: Consider web search for current, real-time, or recent information."
        elif "vector search" in feedback.lower() and "should use" in feedback.lower():
            base_prompt += "\nPRIORITY: Consider vector search for knowledge-base queries and conceptual information."
        elif "graph search" in feedback.lower() and "should use" in feedback.lower():
            base_prompt += "\nPRIORITY: Consider graph search for entity relationships and structured knowledge."

        if "outdated" in feedback.lower() or "current" in feedback.lower():
            base_prompt += "\nIMPORTANT: This query may require current/recent information - consider web search."

        return base_prompt

    def enhance_search_query(self, query, feedback):
        if "not relevant" in feedback.lower():
            return f"Find specific information about: {query}"
        elif "missed information" in feedback.lower():
            return f"Comprehensively search for all aspects of: {query}"
        elif "too shallow" in feedback.lower():
            return f"Find detailed, in-depth information about: {query}"
        elif "current information" in feedback.lower() or "recent" in feedback.lower():
            return f"Latest information about: {query}"
        return query

    def generate_direct_answer_with_feedback(self, query, feedback):
        enhanced_prompt = f"Answer this question: {query}"

        if "more specific" in feedback.lower():
            enhanced_prompt += (
                "\nProvide a specific, detailed answer with concrete examples."
            )
        elif "more comprehensive" in feedback.lower():
            enhanced_prompt += (
                "\nProvide a comprehensive answer covering all relevant aspects."
            )
        elif "missing context" in feedback.lower():
            enhanced_prompt += "\nProvide necessary background context in your answer."

        messages = [
            ChatCompletionSystemMessageParam(
                role="system", content="You are a helpful assistant."
            ),
            ChatCompletionUserMessageParam(role="user", content=enhanced_prompt),
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.3
        )

        return {
            "status": "direct_answer",
            "direct_answer_content": response.choices[0].message.content,
            "feedback_applied": True,
        }


if __name__ == "__main__":
    query_text = "'Query the Imperial College Department of Earth Science and Engineering publication database for publications published in the previous month.', 'Extract the titles and abstracts of all identified publications.', 'For each publication, identify the primary research problem(s) explored based on the title, abstract, and potentially the full text (if accessible).', 'Categorize the identified research problems into distinct themes or areas of focus.', 'Present a consolidated list of publications and their corresponding research problems, grouped by theme.'"
    agent = BrainAgent()
    retrieved_context_info = agent.retrieve(query_text)

    logger.info("\nRetrieved Context Information:")
    if retrieved_context_info["status"] == "tool_executed":
        logger.info(f"  Status: {retrieved_context_info['status']}")
        logger.info(f"  Tool Name: {retrieved_context_info['tool_name']}")
        logger.info(f"  Tool Arguments: {retrieved_context_info['tool_args']}")
        # Truncate long tool output in log
        logger.info(
            f"  Tool Output: {str(retrieved_context_info['tool_output'])[:200]}..."
        )
    elif retrieved_context_info["status"] == "direct_answer":
        logger.info(f"  Status: {retrieved_context_info['status']}")
        # Truncate long direct answer in log
        logger.info(
            f"  Direct Answer: {str(retrieved_context_info['direct_answer_content'])[:200]}..."
        )
    else:  # Error status
        logger.info(f"  Status: {retrieved_context_info['status']}")
        logger.info(
            f"  Message: {retrieved_context_info.get('message', 'No additional message.')}"
        )

    logger.info("\nContext Messages Used for Generation:")
    for msg_item in retrieved_context_info["context_messages_for_generation"]:
        msg_dict = {}
        if hasattr(
            msg_item, "model_dump"
        ):  # Pydantic model (e.g., ChatCompletionMessage)
            msg_dict = msg_item.model_dump(exclude_none=True)
        elif isinstance(
            msg_item, dict
        ):  # TypedDict (e.g., ChatCompletionSystemMessageParam, ChatCompletionToolMessageParam)
            msg_dict = msg_item
        else:
            logger.error(
                f"Warning: Unknown message type in context_messages_for_generation: {type(msg_item)}"
            )
            continue

        role = msg_dict.get("role", "N/A")
        # Ensure content is a string and truncate
        content_str = str(msg_dict.get("content", ""))
        content_display = (
            f"{content_str[:100]}..." if len(content_str) > 100 else content_str
        )

        tool_calls_display = "N/A"
        if "tool_calls" in msg_dict and msg_dict["tool_calls"] is not None:
            # Convert tool_calls to a more readable string format if it's complex
            tool_calls_str = str(msg_dict["tool_calls"])
            tool_calls_display = (
                f"{tool_calls_str[:100]}..."
                if len(tool_calls_str) > 100
                else tool_calls_str
            )

        tool_call_id_value = msg_dict.get(
            "tool_call_id", "N/A"
        )  # Relevant for tool messages (dict form)

        logger.info(
            f"  - Role: {role}, Content: {content_display}, ToolCalls: {tool_calls_display}, ToolCallID: {tool_call_id_value}"
        )
