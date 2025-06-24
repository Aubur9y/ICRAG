import dotenv
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils.model_utils import initialise_llm

dotenv.load_dotenv("config.env")


# Intention understanding
def get_intention(query):
    llm = initialise_llm(model="gpt-4o-mini")

    intention_prompt_template = PromptTemplate.from_template(
        """Your task is to understand the primary intention of the user's query.
        Classify the intention into one of the following categories or provide a concise description if it doesn't fit.
        Categories:
        - Greetings (e.g., "hello", "hi")
        - Specific File Inquiry (e.g., "summarise paper X", "how many lines of code are in notebook Y?"), this is when user knows exactly which files they are looking for
        - Vague File Inquiry (e.g., "which paper can I find information on field A in?", "I remember there is one paper about topic B, do you know which one is it?"), this is when user doesn't know the exact files they are looking for 
        - General Question (e.g., "who are you", "what can you do")
        - Feedback (e.g., "this is confusing", "that is a good summary")
        - Farewell (e.g., "bye", "see you later")
        - Other (if it doesn't fit any of the above, briefly describe the intention)

        User Query: "{query}"
        Identified Intention:
        """
    )

    intention_chain = intention_prompt_template | llm
    try:
        response = intention_chain.invoke({"query": query})
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                return content.strip()
            elif isinstance(content, list):
                return " ".join([str(item) for item in content]).strip()
            else:
                return str(content).strip()
    except Exception as e:
        print(f"Error in get_intention for query '{query}': {e}")
        return "Error: Could not determine intention."


# Agent Setup
def create_intention_agent():
    llm_agent = initialise_llm(model="gpt-4o-mini")

    intention_tool = Tool(
        name="UnderstandUserIntention",
        func=get_intention,
        description="Useful for understanding the user's primary intention on their query. Input should be the user's query string",
    )

    tools = [intention_tool]

    system_message_template = """You are a specialised agegnt tasked with understanding user query intention.
    Your only task is to use the 'UnderstandUserIntention' tool to get the user's intention.
    Your final answer must be the exact string output from the 'UnderstandUserIntention' tool and nothing else.

    You have access to the following tools:
    {tools}

    Use the following format for your thought process:

    Thought: I need to understand the user's query intention. I must use the 'UnderstandUserIntention' tool.
    Action: {tool_names}
    Action Input: The user's query.
    Observation: The identified intention string from the tool.

    Thought: I have received the intention from the tool. My final answer should be this exact intention.
    Final Answer: [The exact intention string from the Observation]
    """

    react_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}\n\n{agent_scratchpad}"),
        ]
    )

    agent = create_react_agent(llm_agent, tools, react_prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
    )
    return agent_executor
