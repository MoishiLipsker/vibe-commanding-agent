"""Action flow node."""

from typing import Dict
from enum import Enum

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import BaseModel, Field

from react_agent.configuration import Configuration
from react_agent.state import State, ActionType
from react_agent.utils import load_chat_model


class ActionClassification(BaseModel):
    """Classification of action type."""
    action_type: ActionType = Field(..., description="The type of action to perform")


action_parser = PydanticOutputParser(pydantic_object=ActionClassification)

SYSTEM_PROMPT = f"""
Classify the action type into one of:
- 'create_entity': Creating a new entity
- 'update_entity': Updating an existing entity that needs to be found first

Expected output JSON structure:
{action_parser.get_format_instructions()}
"""

FEW_SHOT = [
    HumanMessage("Create a new target at coordinates 32.123, 34.456"),
    AIMessage(ActionClassification(action_type=ActionType.CREATE).model_dump_json()),
    
    #HumanMessage("Update the status of the target we found to 'neutralized'"),
    #AIMessage(ActionClassification(action_type=ActionType.UPDATE_WITHOUT_QUERY).model_dump_json()),
    
    HumanMessage("Change the priority of all targets in sector A to high"),
    AIMessage(ActionClassification(action_type=ActionType.UPDATE).model_dump_json())
]


async def check_action(state: State) -> Dict[str, ActionType]:
    """Classify the action type (create, update, or update_without_query).
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        Dict[str, ActionType]: The classified action type.
    """
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model)
    
    # Use trigger_parts.action if available, otherwise use full input
    action_text = state.trigger_parts.action if state.trigger_parts else state.input
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(SYSTEM_PROMPT),
        *FEW_SHOT,
        HumanMessagePromptTemplate.from_template("{input}"),
    ])

    splitter_chain = prompt | model | action_parser

    result = await splitter_chain.ainvoke({"input": action_text})
    
    # Convert from ActionTypeEnum to ActionType
    return {"action_type": ActionType(result.action_type.value)} 