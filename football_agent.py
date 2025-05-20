from langchain.agents import create_react_agent, AgentExecutor
from langchain.chains.question_answering.map_rerank_prompt import output_parser
from langchain_core.tools import Tool
from langchain import hub
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import requests
from pydantic import BaseModel, Field
from typing import List, Dict
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.prompts import PromptTemplate

load_dotenv()

class PlayerSummary(BaseModel):
    summary: str = Field(description="A short summary of the players career")
    clubs: List[str] = Field(description="Clubs the player has played for")
    achievements: List[str] = Field(description="Major career achievements")
    stats: Dict[str, int] = Field(
        description="Overall stats like games, goals and assists"
    )


output_parser = PydanticOutputParser(pydantic_object=PlayerSummary)


def get_player_info(name: str) -> str:
    # return f"Mocked player info {name}. He played for Liverpool FC. Won the champions league. Has over 600 appearances for club and country."
    """Use Tavily to search for soccer player info."""
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": os.getenv("TAVILY_API_KEY"),
            "query": f"{name} soccer career summary",
            "num_results": 3
        }
    )
    results = response.json()
    return "\n\n".join([r["content"] for r in results["results"]])

tools = [
    Tool(
        name="Player Info Search",
        func=get_player_info,
        description="Search the web for a soccer player's career summary, clubs, achievements, and stats.",
    )
]

summary_template  = """You are a football analyst. Given the player's name: {input}, and using available tools,
extract a structured summary of their career in this format:

{format_instructions}

Only output valid JSON. Do not explain."""

react_prompt = hub.pull("hwchase17/react")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# if __name__ == "__main__":
formatted_prompt = PromptTemplate.from_template(summary_template ).partial(
    format_instructions=output_parser.get_format_instructions()
)

prompt_runnable = RunnableLambda(lambda x: formatted_prompt.invoke(x))

chain = prompt_runnable | llm | output_parser

# player_name = "Steven Gerrard"

# result = chain.invoke({"input": player_name})

# print(result)