"""Trigger flow node."""
from typing import Dict

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import BaseModel, Field

from react_agent.configuration import Configuration
from react_agent.state import State, TriggerParts
from react_agent.utils import load_chat_model


class TriggerQuery(BaseModel):
    query: str = Field(..., description="The condition that should be checked")
    action: str = Field(..., description="What should happen when the condition is met")

trigger_parser = PydanticOutputParser(pydantic_object=TriggerQuery)

SYSTEM_PROMPT = f"""
Split the trigger request into two natural language parts:
1. The query part - what condition should be checked
2. The action part - what should happen when the condition is met

Expected output JSON structure:
{trigger_parser.get_format_instructions()}
"""

FEW_SHOT = [
    HumanMessage("When a car enters the parking lot, send me a notification"),
    AIMessage(TriggerQuery(
        query="a car enters the parking lot",
        action="send a notification to me about a car entering the parking lot"
    ).model_dump_json()),
    
    HumanMessage("If temperature exceeds 30 degrees, turn on the air conditioning"),
    AIMessage(TriggerQuery(
        query="temperature exceeds 30 degrees",
        action="turn on the air conditioning when temperature exceeds 30 degrees"
    ).model_dump_json()),

    HumanMessage("When someone mentions 'urgent' in the chat, send an alert to the admin"),
    AIMessage(TriggerQuery(
        query="someone mentions 'urgent' in the chat",
        action="send an alert to the admin about urgent message in the chat"
    ).model_dump_json()),
]


async def trigger_extract(state: State) -> Dict[str, TriggerParts]:
    """Split trigger request into natural language query and action parts.
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        Dict[str, TriggerParts]: Natural language query and action parts.
    """
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(SYSTEM_PROMPT),
        *FEW_SHOT,
        HumanMessagePromptTemplate.from_template("{input}"),
    ])

    splitter_chain = prompt | model | trigger_parser

    result = await splitter_chain.ainvoke({"input": state.input})
    
    return {"trigger_parts": TriggerParts(query=result.query, action=result.action)}
    