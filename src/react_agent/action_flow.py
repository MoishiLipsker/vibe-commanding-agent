"""A simple chatbot."""

from typing import Dict, List, Optional, Any, Sequence
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import Annotated
from langgraph.graph import StateGraph, END

SYSTEM_PROMPT = """
You are a friendly, curious, geeky AI.
"""

class InputState(BaseModel):
    """Input state for the workflow."""
    #messages: List[Dict[str, str]]
    input: Optional[str] = None
    position: Optional[dict] = None
    selected_entity: Optional[dict] = None
    #entities: Optional[Dict] = None
    #schemas: Optional[Dict] = None

class OutputState(BaseModel):
    """output state for the workflow."""
    response: Optional[str] = Field(None, description="Response from the API call")

class State(BaseModel):
    """State for the workflow."""
    input: str = Field(..., description="The user's input")


    class Config:
        arbitrary_types_allowed = True


def process_input(state: State) -> Dict[str, str]:
    """Process the input."""
    return {"response": "Hello, world!"}


def create_workflow() -> StateGraph:
    """Create the workflow graph.
    
    Returns:
        StateGraph: The workflow graph.
    """
    # Create the graph
    workflow = StateGraph(State, input=InputState, output=OutputState)
    
    # Add nodes
    workflow.add_node("process_input", process_input)
    
    # Add edges for classify_input
    workflow.set_entry_point("process_input")
    workflow.add_edge("process_input", END)

    
    return workflow

# Create the workflow graph
workflow = create_workflow()
# workflow.validate()

action_flow = workflow.compile()