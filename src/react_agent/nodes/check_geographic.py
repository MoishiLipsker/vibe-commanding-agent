"""Geographic check node."""

from typing import Dict
from enum import Enum

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import BaseModel, Field

from react_agent.configuration import Configuration
from react_agent.state import State
from react_agent.utils import load_chat_model


class GeographicType(str, Enum):
    """Available geographic types."""
    GEOGRAPHIC = "geographic"
    NON_GEOGRAPHIC = "non_geographic"


class GeographicClassification(BaseModel):
    """Geographic type classification result."""
    is_geographic: bool = Field(..., description="Whether the request is geographic or not")
    explanation: str = Field(..., description="Explanation for why this classification was chosen")


geo_parser = PydanticOutputParser(pydantic_object=GeographicClassification)

SYSTEM_PROMPT = f"""
Determine if the user's request is geographic or non-geographic.
A request is considered geographic if it involves:
- Location-based conditions or actions
- Spatial relationships between entities
- Geographic boundaries or areas
- Movement or positioning in space
- Geographic coordinates or locations

Provide both the classification and an explanation for your choice.

Expected output JSON structure:
{geo_parser.get_format_instructions()}
"""

FEW_SHOT = [
    HumanMessage("When a car crosses the eastern border, send an alert"),
    AIMessage(GeographicClassification(
        is_geographic=True,
        explanation="This is geographic because it involves a spatial condition (crossing a border) and location-based action"
    ).model_dump_json()),
    
    HumanMessage("Send a notification when the temperature exceeds 30 degrees"),
    AIMessage(GeographicClassification(
        is_geographic=False,
        explanation="This is non-geographic because it's based on a temperature condition, not a spatial or location-based condition"
    ).model_dump_json()),
    
    HumanMessage("Alert me when any vehicle enters sector A"),
    AIMessage(GeographicClassification(
        is_geographic=True,
        explanation="This is geographic because it involves a spatial condition (entering a specific sector) and location-based monitoring"
    ).model_dump_json())
]


async def check_geographic(state: State) -> Dict[str, bool]:
    """Check if the request is geographic or not.
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        Dict[str, bool]: Whether the request is geographic or not.
    """
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model)
    
    # Use trigger_parts.query if available, otherwise use full input
    query_text = state.trigger_parts.query if state.trigger_parts else state.input
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(SYSTEM_PROMPT),
        *FEW_SHOT,
        HumanMessagePromptTemplate.from_template("{input}"),
    ])

    splitter_chain = prompt | model | geo_parser

    result = await splitter_chain.ainvoke({"input": query_text})
    
    return {"is_geographic": result.is_geographic} 