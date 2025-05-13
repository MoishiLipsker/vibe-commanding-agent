"""Input classification node."""

from typing import Dict
from enum import Enum

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import BaseModel, Field

from react_agent.configuration import Configuration
from react_agent.state import State
from react_agent.utils import load_chat_model


class FlowType(str, Enum):
    """Available flow types."""
    TRIGGER = "trigger"
    QUERY = "query"
    ACTION = "action"


class FlowClassification(BaseModel):
    """Flow type classification result."""
    flow_type: FlowType = Field(..., description="The type of flow (trigger, query, or action)")
    explanation: str = Field(..., description="Explanation for why this flow type was chosen")


flow_parser = PydanticOutputParser(pydantic_object=FlowClassification)

SYSTEM_PROMPT = f"""
Classify the user's request into one of the following flow types:
- trigger: A request that sets up a condition and an action to perform when the condition is met
- query: A request to find or get information about existing entities
- action: A request to create or modify entities directly

Provide both the flow type and an explanation for your choice.

Expected output JSON structure:
{flow_parser.get_format_instructions()}
"""

FEW_SHOT = [
    HumanMessage("When a car enters the parking lot, send me a notification"),
    AIMessage(FlowClassification(
        flow_type=FlowType.TRIGGER,
        explanation="This is a trigger because it sets up a condition (car entering) and an action to perform (send notification) when that condition is met"
    ).model_dump_json()),
    
    HumanMessage("Show me all targets in sector A"),
    AIMessage(FlowClassification(
        flow_type=FlowType.QUERY,
        explanation="This is a query because it's asking to find and display information about existing targets"
    ).model_dump_json()),
    
    HumanMessage("Create a new target at coordinates 32.123, 34.456"),
    AIMessage(FlowClassification(
        flow_type=FlowType.ACTION,
        explanation="This is an action because it's directly requesting to create a new entity"
    ).model_dump_json())
]


async def classify_input(state: State) -> Dict[str, str]:
    """Classify the user's input to determine the flow type.
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        Dict[str, str]: The flow type to use.
    """
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(SYSTEM_PROMPT),
        *FEW_SHOT,
        HumanMessagePromptTemplate.from_template("{input}"),
    ])

    splitter_chain = prompt | model | flow_parser

    result = await splitter_chain.ainvoke({"input": state.input})
    
    return {"flow_type": result.flow_type.value, "messages": [HumanMessage(state.input)]} 