"""Geographic query processing node."""
import os

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, Runnable
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from react_agent.configuration import Configuration
from react_agent.state import State
from react_agent.utils import load_chat_model


class GeographicQuery(BaseModel):
    source: str = Field(..., description="The entity that performs the action")
    target: str = Field(..., description="The entity being monitored or affected")
    relation: str = Field(..., description="The interaction or relationship between the source and target")
geo_rule_Splitter_parser = PydanticOutputParser(pydantic_object=GeographicQuery)

SYSTEM_PROMPT = f"""
Extract GeoRule elements from the user request.
Expected output JSON structure:
{geo_rule_Splitter_parser.get_format_instructions()}

"""

FEW_SHOT = [
    HumanMessage("Car with yellow color that crosses the east general border"),
    AIMessage(GeographicQuery(source="Car with yellow color", target="East general border", relation="that crosses").model_dump_json()),

    HumanMessage("hunter with name Jon is crossing the sea in north"),
    AIMessage(GeographicQuery(source="hunter with name Jon", target="sea in north", relation="crossing").model_dump_json()),
]

async def geographic_extract(state: State) -> GeographicQuery:
    """Split geographic queries into source and destination parts.
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        GeographicQuery: Structured object containing source and destination parts.
    """
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model)
    
    prompt = ChatPromptTemplate.from_messages([
            SystemMessage(SYSTEM_PROMPT),
            *FEW_SHOT,
            HumanMessagePromptTemplate.from_template("{input}"),
    ])

    splitter_chain = prompt | model | geo_rule_Splitter_parser

    result = await splitter_chain.ainvoke({"input": state.trigger_parts.query})
    
    # Parse the response into a GeographicQuery object
    return {"raw_source_query": result.source, "raw_destination_query": result.target}

