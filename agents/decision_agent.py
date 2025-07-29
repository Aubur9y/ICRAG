from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import requests
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from utils.model_service import chat_with_model, chat_with_model_thinking
from utils.prompt_templates import DECISION_AGENT_GENERATION_PROMPT

load_dotenv()


class DecisionAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.system_prompt = DECISION_AGENT_GENERATION_PROMPT

    # take in the contexts generated from last step
    def generate(self, user_prompt, confidence_level="high"):
        if confidence_level == "low":
            system_prompt = (
                self.system_prompt
                + "\nIMPORTANT: The retrieved information has low relevance to the query. Express uncertainty clearly and avoid making definitive statements. If you cannot provide a reliable answer based on the context, state this clearly."
            )
        else:
            system_prompt = self.system_prompt

        # Using OpenAI API
        # messages = [
        #     ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        #     ChatCompletionUserMessageParam(role="user", content=user_prompt),
        # ]
        # response = self.client.chat.completions.create(
        #     model=self.model, messages=messages
        # )

        # Using OpenWebUI API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        t0 = time.time()
        response = chat_with_model(
            messages=messages,
        )

        print(f"Time taken for DecisionAgent: {time.time() - t0:.2f}s")
        return response


if __name__ == "__main__":
    context = """
        Context 1: Smart City Technologies
        Smart cities integrate Internet of Things (IoT) devices, sensors, and data analytics platforms to monitor and manage urban resources efficiently. Key components include smart grids that optimize electricity distribution, intelligent transportation systems that reduce congestion by 15-30%, and connected infrastructure that enables real-time decision making. Studies show that implementing comprehensive IoT networks can reduce energy consumption in urban areas by up to 20% and cut operational costs by approximately $12 billion annually across major metropolitan areas.
        
        Context 2: Environmental Impact Research
        Research from the Environmental Systems Journal (2023) indicates that smart city technologies can reduce carbon emissions by 16-27% when fully implemented. However, the manufacturing and disposal of IoT devices contribute to electronic waste, with an estimated 350,000 tons of additional e-waste generated annually due to smart city implementations. The energy consumption of data centers supporting smart city operations increased by 22% between 2020-2023, potentially offsetting some environmental benefits unless powered by renewable sources.
        
        Context 3: Case Study - Singapore
        Singapore's Smart Nation initiative, launched in 2014, has successfully deployed over 110,000 sensors monitoring everything from traffic to public safety. Their Virtual Singapore digital twin project allows for detailed urban planning simulations. Quantifiable results include a 12% reduction in traffic congestion, 15% improvement in emergency response times, and 8% reduction in energy consumption across public buildings. Public satisfaction surveys show 72% of residents believe the smart city features have improved quality of life.
        
        Context 4: Digital Divide Concerns
        Implementation of smart city technologies has exacerbated digital divides in multiple regions. A 2022 study across 15 major cities found that low-income neighborhoods were 35% less likely to benefit from smart city innovations than affluent areas. Additionally, 62% of elderly residents reported difficulty accessing digital services that replaced traditional options. Privacy concerns remain significant, with 58% of urban residents expressing concern about surveillance aspects of smart city technologies according to a 2023 Pew Research survey.
        
        Context 5: Future Developments
        Emerging technologies expected to reshape smart cities include 6G networks (projected deployment: 2028-2030), quantum computing applications for traffic optimization, and blockchain-based systems for secure data sharing between city services. The global smart city market is projected to grow from $457 billion in 2023 to approximately $873 billion by 2030. Climate adaptation is becoming a central focus, with 78% of new smart city projects incorporating climate resilience elements.
    """

    query = "Develop a comprehensive strategy for mid-sized cities (population 1-3 million) to implement smart city technologies over the next decade, addressing potential environmental impacts, ensuring equitable access across socioeconomic groups, and maximizing cost-effectiveness. Include specific technology recommendations, implementation timelines, methods to measure success, and approaches to mitigate identified challenges. How should these cities balance immediate benefits against long-term sustainability concerns?"

    initial_prompt = f"""
    Answer the question "{query}" based on the following context:
    context: {context}
    
    Please provide accurate response based on the above contexts.
    """

    agent = DecisionAgent()
    response = agent.generate(initial_prompt)
    print(response)
