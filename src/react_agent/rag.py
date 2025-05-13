"""A simple chatbot."""

from langgraph.prebuilt import create_react_agent
from langchain.tools.retriever import create_retriever_tool

from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings

client = QdrantClient(":memory:")

# client.create_collection(
#     collection_name="demo_collection",
#     vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
# )

retriever = vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=OpenAIEmbeddings(),
).as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_my_texts",
    "Retrieve texts stored in the Qdrant collection",
)

tools = [retriever_tool]

rag = create_react_agent(
      "anthropic:claude-3-5-haiku-latest",
      tools=tools,
      prompt="You are a friendly, curious, geeky AI.",
)