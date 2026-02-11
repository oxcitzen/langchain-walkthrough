from langchain.agents.middleware import dynamic_prompt, ModelRequest
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

@dynamic_prompt
def user_role_prompt(request: ModelRequest):
    
    ctx = request.runtime.context or {}
    role = ctx.get("user_role", "beginner")
    if role == "child":
        return "Explain everything as if you are talking to a 5-year-old."
    return "Provide technical responses."

agent = create_agent(
    model="gpt-4o-mini",
    middleware=[user_role_prompt]
)

# Example call
res = agent.invoke(
    {"messages": [{"role": "user", "content": "What is PCA?"}]},
    context={"user_role": "child"}
    
)
print(res["messages"][-1].content)