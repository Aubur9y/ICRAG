import pytest
from utils.user_query_understanding import get_intention, create_intention_agent

sample_queries = [
    ("Hello there!", "Greetings"),
    ("Can you summarise in paper A?", "Specific File Inquiry"),
    ("What are you?", "General Question"),
    ("Bye bye.", "Farewell"),
    ("This is really confusing, can you elaborate?", "Feedback"),
    ("You are very helpful!", "Feedback"),
]

@pytest.mark.parametrize("query,expected_intention", sample_queries)
def test_get_intention_classification(query, expected_intention):
    intention = get_intention(query)
    assert expected_intention in intention

def test_create_intention_agent_runs():
    agent = create_intention_agent()
    response = agent.invoke({
        "input": "What can you do?",
        "chat_history": []
    })
    assert "output" in response
    assert isinstance(response['output'], str)