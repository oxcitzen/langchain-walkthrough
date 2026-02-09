import requests
from dataclasses import dataclass
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime
from typing import Optional

load_dotenv()

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    summary: str
    temp_celsius: Optional[float] = None
    humidity: Optional[float] = None

# Tools can access 'runtime context' to look up user data
@tool('get_weather', description="Return weather information for a given city.", return_direct=False)
def get_weather(city: str):
    """Return weather information for a given city. (Public API)"""
    url = f"https://wttr.in/{city}?format=j1"
    response = requests.get(url)
    return response.json()

@tool('locate_user', description="Look up a user's city based on context.", return_direct=False)
def locate_user(runtime: ToolRuntime[Context]):
    """Look up a user's city based on context.""" 
    user_id = runtime.context.user_id
    db = {"ABC123": "Tokyo", "XYZ456": "London"}
    return db.get(user_id, "Unknown")
    
    
# checkpoint, to remember conversation
memory = InMemorySaver()

agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather, locate_user],
    system_prompt="You are a helpful weather assistant that provides weather information for a given city.",
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=memory
)

config = {"configurable": {"thread_id": "1"}}
context = Context(user_id="ABC123")

res = agent.invoke(
    {"messages": [{"role": "user", "content": "How is the weather?"}]},
    config=config,
    context=context
)

print(type(res))
#print(res['structured_response'])
print(res['structured_response'].summary)
print(res['structured_response'].temp_celsius)
#print(f"Summary: {res.structured_response.summary}")
#print(f"Summary: {res.structured_response.temp_celsius}")


# Start a new thread for new conversation
#config = {"configurable": {"thread_id": "2"}}

res = agent.invoke(
    {"messages": [{"role": "user", "content": "And is this a usual weather for this season?"}]},
    config=config,
    context=context
)

print(res['structured_response'].summary)
