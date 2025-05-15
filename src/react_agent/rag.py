"""A simple chatbot."""

from langgraph.prebuilt import create_react_agent
from langchain.tools.retriever import create_retriever_tool

from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings

client = QdrantClient(url="http://10.2.3.9:6333")


retriever = vector_store = QdrantVectorStore(
     client=client,
     collection_name="entities",
     embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
).as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_my_texts",
    "Retrieve relevant entities from the vector database to answer investigative questions.",
)

tools = [retriever_tool]

rag = create_react_agent(
      "openai:gpt-4o",
      tools=tools,
      prompt="You are an expert AI assistant. Your purpose is to answer investigative questions professionally, accurately, and reliably. Use the information from the retrieved documents to provide comprehensive answers. If necessary, present information in Markdown tables. Remember, your answers are used in a security product, so accuracy and reliability are paramount.",
)