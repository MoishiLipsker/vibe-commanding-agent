"""Action flow implementation with LangGraph."""

from typing import Dict, List, Optional, Any, Sequence
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import Annotated
from langgraph.graph import StateGraph, END

from react_agent.nodes.action_flow_summary import ActionFlowSummaryHandler, ActionFlowEvent

SYSTEM_PROMPT = """
You are a friendly, curious, geeky AI that helps explain system events and actions.
"""

class InputState(BaseModel):
    """Input state for the workflow."""
    trigger: Optional[dict] = None
    entity: Optional[dict] = None

class OutputState(BaseModel):
    """Output state for the workflow."""
    response: Optional[str] = Field(None, description="Explanation of what happened")

class State(BaseModel):
    """State for the workflow."""
    input: str = Field(..., description="The user's input")
    trigger: Optional[dict] = None
    entities: Optional[list] = None
    response: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


async def explain_event(state: State) -> Dict[str, Any]:
    """Generate explanation for the event using ActionFlowSummaryHandler."""
    handler = ActionFlowSummaryHandler()
    
    event = ActionFlowEvent(
        trigger_query=state.trigger["rawTrigger"],
        entity_data=state.entities,
        planned_action=state.trigger["rawAction"]
    )
    
    response = ""
    async for chunk in handler.explain_event(event):
        response = chunk["explanaresponsetion"]
    
    return {
        "response": response,
    }


def create_workflow() -> StateGraph:
    """Create the workflow graph.
    
    Returns
    -------
    StateGraph
        The workflow graph with process and explain nodes.
    """
    # Create the graph
    workflow = StateGraph(State, input=InputState, output=OutputState)
    
    # Add nodes
    workflow.add_node("explain_event", explain_event)
    
    # Set the flow
    workflow.set_entry_point("explain_event")
    
    return workflow

# Create the workflow graph
workflow = create_workflow()
action_flow = workflow.compile()