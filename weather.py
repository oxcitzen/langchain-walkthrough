import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

@tool
def get_weather(city: str):
    """Return weather information for a given city. (Public API)"""
    url = f"https://wttr.in/{city}?format=j1"
    response = requests.get(url)
    return response.json()

# Create the agent
agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant that provides weather information for a given city."
)

# Invoke the agent
# The API is doing the heavy lifting with KL interpret as Kuala Lumpur
response = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather like in KL?"}]
})

print(response["messages"][-1].content)