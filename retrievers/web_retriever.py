from duckduckgo_search import DDGS
from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

class WebRetrieverAgent:
    def __init__(self, top_k=5):
        self.top_k = top_k

    def retrieve(self, query):
        # results = []
        # with DDGS() as ddgs:
        #     for r in ddgs.text(query, safesearch="Moderate", max_results=self.top_k):
        #         results.append({
        #             "title": r.get("title"),
        #             "snippet": r.get("body"),
        #             "link": r.get("href")
        #         })
        # return results

        tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        response = tavily_client.search(query)
        return response

if __name__ == "__main__":
    agent = WebRetrieverAgent(top_k=3)
    query = "What is the weather like in London today?"
    results = agent.retrieve(query)

    # for res in results:
    #     print(f"\nTitle: {res['title']}\nSnippet: {res['snippet']}\nLink: {res['link']}")

    print(results['results'][0]['content'])