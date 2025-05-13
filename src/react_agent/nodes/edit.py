"""Edit flow node."""

from typing import Dict, cast
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate

from react_agent.configuration import Configuration
from react_agent.state import State, QueryParams
from react_agent.utils import load_chat_model

async def action_extract(state: State) -> Dict[str, str]:
    """Process the edit part into structured query parameters.
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        Dict[str, QueryParams]: Structured query parameters.
    """
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model)
    
    # Use the appropriate action text based on the flow
    action_text = state.trigger_parts.action if state.trigger_parts else state.input

    # Create a structured prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that converts an action text into a natural language query to find the entity.
        The query should be in the form of a natural language request to find the specific entity.
        For example:
        - From "update the goal named 'Complete project documentation' to high priority" extract "find the goal named 'Complete project documentation'"
        - From "change the user with email 'john.doe@example.com' display name" extract "find the user with email 'john.doe@example.com'"
        - From "modify the task called 'Implement login page' status" extract "find the task called 'Implement login page'"
        - From "set the project titled 'Q4 Website Redesign' deadline" extract "find the project titled 'Q4 Website Redesign'"
        
        Return only the natural language query without any additional text."""),
        ("human", "{action_text}")
    ])
    
    # Create the chain
    chain = prompt | model
    
    # Run the chain
    result = await chain.ainvoke({"action_text": action_text})
    
    # Extract the query segment from the response
    query_segment = result.content.strip()
    
    return {"raw_query_params_for_edit": query_segment}