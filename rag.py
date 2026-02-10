from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent
from langchain_core.tools import create_retriever_tool
from dotenv import load_dotenv

load_dotenv()

texts = ["I love apples.", "I hate bananas.", "I enjoy oranges.", "I love grapes", "Apple makes very good computers", "Apple is an innovative company"]

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = FAISS.from_texts(texts, embeddings)

'''
Very different result with 'My favorite fruit is apple.' vs 'Apple is my favorite fruit.'

For first statement, the fruits have higher ranking than the Apple company.
For second staement, the Apple company ranks higher in terms of similarity than other fruits.

Embeddings can't differentiate the nuances and prioritize semantic proximity + phrasing similarity, not category purity.
'''

print(vector_store.similarity_search('My favorite fruit is apple.', k = 6))

# print(vector_store.similarity_search('Apple is my favorite fruit.', k = 6))

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
retriever_tool = create_retriever_tool(
    retriever, 
    "fruit_search", 
    "Search for user fruit preferences."
)

agent = create_agent(model="gpt-4o-mini", tools=[retriever_tool])
res = agent.invoke({"messages": [{"role": "user", "content": "Which fruits do I like?"}]})

# Expected result with correct retrieval
print(res["messages"][-1].content)