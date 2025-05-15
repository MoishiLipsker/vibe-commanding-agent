"""Action flow implementation with LangGraph."""

from typing import Dict, List, Optional, Any, Sequence
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from react_agent.nodes.action_flow_summary import ActionFlowSummaryHandler, ActionFlowEvent
from react_agent.configuration import Configuration
from react_agent.utils import load_chat_model
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
import json

SYSTEM_PROMPT = """
You are a friendly, curious, geeky AI that helps explain system events and actions.
"""

class InputState(BaseModel):
    """Input state for the workflow."""
    trigger: Optional[dict] = None
    entities: Optional[list] = None

class OutputState(BaseModel):
    """Output state for the workflow."""
    response: Optional[str] = Field(None, description="Explanation of what happened")
    refined_action: Optional[Dict[str, Any]] = Field(None, description="The action payload, potentially refined by an LLM using entity data. If no refinement was needed, this will be the original action payload.")

class State(BaseModel):
    """State for the workflow."""
    input: str = Field(..., description="The user's input")
    trigger: Optional[dict] = None
    entities: Optional[list] = None
    response: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

class RefinedActionOutput(BaseModel):
    """Defines the expected JSON structure for the refined action payload from the LLM."""
    refined_action_payload: Dict[str, Any] = Field(description="The complete action payload, refined with information from entities and trigger context.")

refined_action_parser = PydanticOutputParser(pydantic_object=RefinedActionOutput)

async def _refine_action_with_llm(
    action_to_refine: Dict[str, Any], 
    entities: Optional[List[Any]],
    trigger_query_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Refines the action payload using an LLM, incorporating entity data based on trigger intent.

    Parameters
    ----------
    action_to_refine : Dict[str, Any]
        The original action dictionary (e.g., state.trigger["rawAction"]).
    entities : Optional[List[Any]]
        A list of entity dictionaries available from the trigger. 
        Each entity is expected to be a dict, ideally with 'id', 'type', and 'data' keys.
    trigger_query_context : Dict[str, Any]
        The raw trigger query (e.g., state.trigger["rawTrigger"]) for overall context,
        especially the natural language query part.

    Returns
    -------
    Dict[str, Any]
        The refined action dictionary.
    """
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model)


    # Serialize complex objects for the prompt to ensure they are well-formatted strings
    action_to_refine_str = json.dumps(action_to_refine, indent=2)
    entities_str = json.dumps(entities, indent=2) if entities else "[]"

    REFINE_ACTION_SYSTEM_PROMPT = f"""
You are an AI assistant tasked with refining an action payload based on a user's trigger query and a list of relevant entities.
Your goal is to update the provided 'current_action_payload' by intelligently incorporating information from the 'available_entities'.
The 'trigger_context' (especially the natural language part) will guide you on *how* to use the entities.

For example, if the trigger_context mentions "set the force as the executing unit" and the 'available_entities' list contains an entity of type 'force',
you should find the ID of that 'force' entity and place it into an 'executing_unit' field (or a similarly named field) within the 'current_action_payload'.

Follow these critical guidelines:
1.  **Primary Task**: Modify the 'current_action_payload' by filling in placeholders or adding/updating fields that require information from 'available_entities', as guided by the 'trigger_context'.
2.  **Entity Information**: Entities in 'available_entities' will typically have an 'id', 'type', and 'data' (containing specific fields). Use the 'id' when a reference to an entity is needed (e.g., for an 'executing_unit' field).
3.  **Maintain Structure**: Preserve the original structure of 'current_action_payload' as much as possible. Only add or modify fields that are explicitly implied by the trigger context and can be derived from the entities. Do not remove fields unless instructed by context.
4.  **Return Format**: You MUST return a single JSON object. This JSON object must conform to the following structure:
    {{
        "refined_action_payload": {{... a dictionary representing the complete, updated action payload ...}}
    }}
    The 'refined_action_payload' key's value should be the entire action payload, including both original unmodified fields and the new/updated fields.
5.  **No Direct Explanation**: Do not add any conversational text or explanation in your output. Only the JSON object as specified.
6.  **Handle Missing Information**: If necessary information from entities is missing, or the trigger context is unclear for a specific modification, make a best effort or leave that part of the payload as is. Do not invent information not present in the inputs.

Here is the information you will work with:

Current Action Payload (to be refined):
```json
{action_to_refine_str}
```

Available Entities (use these to find information like IDs):
```json
{entities_str}
```

Trigger Context:
{trigger_query_context}

Expected output format instructions:
{refined_action_parser.get_format_instructions()}

Now, generate the refined action payload based on all the provided information.
"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=REFINE_ACTION_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("Refine the action payload based on the provided current_action_payload, available_entities, and trigger_context.")
    ])

    chain = prompt | model | refined_action_parser

    try:
        # The input to invoke can be an empty dict if the template doesn't use any {{input_variable}} from HumanMessage
        # or we can pass the structured data if we decide to reference them via {{variables}} in HumanMessage.
        # For this setup, the system prompt ingests all data directly.
        llm_result = await chain.ainvoke({}) 
        return llm_result.refined_action_payload
    except Exception as e:
        print(f"Error refining action payload with LLM: {e}")
        print(f"Original action payload: {action_to_refine}")
        # Fallback to original action if LLM refinement fails
        return action_to_refine


async def explain_event(state: State) -> Dict[str, Any]:
    """Generate explanation for the event using ActionFlowSummaryHandler."""
    payload = state.trigger["actions"][0]["payload"]
    raw_action = state.trigger["rawAction"]
    refined_action = await _refine_action_with_llm(
        action_to_refine=payload,
        entities=state.entities,
        trigger_query_context=raw_action
    )

    handler = ActionFlowSummaryHandler()
    
    event = ActionFlowEvent(
        trigger_query=state.trigger["rawTrigger"],
        entity_data=state.entities,
        planned_action=raw_action
    )
    
    response = ""
    async for chunk in handler.explain_event(event):
        response = chunk["explanation"]
    
    return {
        "response": response,
        "refined_action": refined_action
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