import os
import json
import logging
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionMessage
)
from dotenv import load_dotenv
from retrievers.vector_retriever import VectorRetrieverAgent
from retrievers.web_retriever import WebRetrieverAgent
from utils.prompt_templates import BRAIN_QUERY_TOOL_USE_SYSTEM_PROMPT

load_dotenv()
logging.basicConfig(
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class BrainAgent:
    MODEL_NAME = "gpt-4o"
    VECTOR_DB_COLLECTION_NAME = "ipynb_embeddings"

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
                            "query": {"type": "string", "description": "The query to search"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Use this for real-time or current events",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The query to search"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        self.system_prompt_content = BRAIN_QUERY_TOOL_USE_SYSTEM_PROMPT
        self.vector_retriever = VectorRetrieverAgent(collection_name=self.VECTOR_DB_COLLECTION_NAME)
        self.web_retriever = WebRetrieverAgent()

    def _decide_tool(self, query) -> ChatCompletionMessage:
        messages = [
            ChatCompletionSystemMessageParam(role="system", content=self.system_prompt_content),
            ChatCompletionUserMessageParam(role="user", content=query)
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=self.tool_specs,
            tool_choice="auto"
        )
        return response.choices[0].message

    def _execute_tool(self, tool_name, tool_args_dict):
        tool_query = ""
        if isinstance(tool_args_dict, dict) and "query" in tool_args_dict:
            tool_query = tool_args_dict["query"]
        elif isinstance(tool_args_dict, str):
            tool_query = tool_args_dict
        else:
            tool_query = str(tool_args_dict)

        result = None
        if tool_name == "vector_search":
            result = self.vector_retriever.retrieve(tool_query)
        elif tool_name == "web_search":
            search_result = self.web_retriever.retrieve(tool_query).get('results', [])
            if not search_result:
                result = "No web search results found."
            else:
                content_parts = []
                for i, res in enumerate(search_result[:3]):
                    content_parts.append(f"Source {i + 1}: {res.get('content', 'No content available')}")
                result = "\n\n".join(content_parts)
        else:
            result = "Unknown tool call."

        logger.info(f"Tool '{tool_name}' returns: {result}")
        return result

    def retrieve(self, query) -> dict:
        assistant_initial_message = self._decide_tool(query)

        context_messages = [
            ChatCompletionSystemMessageParam(role="system", content=self.system_prompt_content),
            ChatCompletionUserMessageParam(role="user", content=query),
            assistant_initial_message
        ]

        if hasattr(assistant_initial_message, "tool_calls") and assistant_initial_message.tool_calls:
            tool_call = assistant_initial_message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments

            try:
                tool_args_dict = json.loads(tool_args_str)
            except json.JSONDecodeError:
                logger.error(f"Error: Decoding tool argument failed: {tool_args_str}")
                return {
                    "status": "error",
                    "message": "Decoding tool argument failed.",
                    "context_messages_for_generation": context_messages
                }

            logger.info(f"Tool decision: use tool '{tool_name}', arguments: {tool_args_str}")

            tool_output_content = self._execute_tool(tool_name, tool_args_dict)

            if isinstance(tool_output_content, dict) and tool_output_content.get("status") == "no_relevant_information_found":
                return {
                    "status": "no_relevant_information",
                    "message": "No relevant information found in knowledge base.",
                    "context_messages_for_generation": context_messages
                }

            tool_result_message = ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=tool_call.id,
                content=json.dumps(tool_output_content) if not isinstance(tool_output_content, str) else tool_output_content
            )
            context_messages.append(tool_result_message)

            return {
                "status": "tool_executed",
                "tool_name": tool_name,
                "tool_args": tool_args_dict,
                "tool_output": tool_output_content,
                "context_messages_for_generation": context_messages
            }
        else:
            logger.info("No tools used.")
            return {
                "status": "direct_answer",
                "direct_answer_content": assistant_initial_message.content,
                "context_messages_for_generation": context_messages
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
        logger.info(f"  Tool Output: {str(retrieved_context_info['tool_output'])[:200]}...")
    elif retrieved_context_info["status"] == "direct_answer":
        logger.info(f"  Status: {retrieved_context_info['status']}")
        # Truncate long direct answer in log
        logger.info(f"  Direct Answer: {str(retrieved_context_info['direct_answer_content'])[:200]}...")
    else:  # Error status
        logger.info(f"  Status: {retrieved_context_info['status']}")
        logger.info(f"  Message: {retrieved_context_info.get('message', 'No additional message.')}")

    logger.info("\nContext Messages Used for Generation:")
    for msg_item in retrieved_context_info["context_messages_for_generation"]:
        msg_dict = {}
        if hasattr(msg_item, 'model_dump'):  # Pydantic model (e.g., ChatCompletionMessage)
            msg_dict = msg_item.model_dump(exclude_none=True)
        elif isinstance(msg_item, dict):  # TypedDict (e.g., ChatCompletionSystemMessageParam, ChatCompletionToolMessageParam)
            msg_dict = msg_item
        else:
            logger.error(f"Warning: Unknown message type in context_messages_for_generation: {type(msg_item)}")
            continue

        role = msg_dict.get('role', 'N/A')
        # Ensure content is a string and truncate
        content_str = str(msg_dict.get('content', ''))
        content_display = f"{content_str[:100]}..." if len(content_str) > 100 else content_str

        tool_calls_display = 'N/A'
        if 'tool_calls' in msg_dict and msg_dict['tool_calls'] is not None:
            # Convert tool_calls to a more readable string format if it's complex
            tool_calls_str = str(msg_dict['tool_calls'])
            tool_calls_display = f"{tool_calls_str[:100]}..." if len(tool_calls_str) > 100 else tool_calls_str


        tool_call_id_value = msg_dict.get('tool_call_id', 'N/A') # Relevant for tool messages (dict form)

        logger.info(
            f"  - Role: {role}, Content: {content_display}, ToolCalls: {tool_calls_display}, ToolCallID: {tool_call_id_value}")
