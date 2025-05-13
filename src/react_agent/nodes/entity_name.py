"""Query flow nodes."""

from typing import Dict, cast, List
from enum import Enum
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import BaseModel, Field

from react_agent.configuration import Configuration
from react_agent.utils import load_chat_model
from react_agent.state import State


class EntityType(str, Enum):
    """Available entity types."""
    TARGET = "Target"
    ISR_TASK = "ISRTask"
    ARROW = "Arrow"
    FRIENDLY_FORCE = "FriendlyForce"
    PHASE_LINE = "PhaseLine"
    CASEVAC_TASK = "CASEVACTask"
    FIRE_MISSION = "FireMission"
    MESSAGE = "Message"


class EntityExtraction(BaseModel):
    """Entity type extraction result."""
    entity_type: EntityType = Field(..., description="The extracted entity type")
    explanation: str = Field(..., description="Explanation for why this entity type was chosen")


entity_parser = PydanticOutputParser(pydantic_object=EntityExtraction)

SYSTEM_PROMPT = f"""
Identify which entity type the user is referring to from this list: {[e.value for e in EntityType]}.
Provide both the entity type and an explanation for your choice.

Expected output JSON structure:
{entity_parser.get_format_instructions()}
"""

FEW_SHOT = [
    HumanMessage("Create a new target at coordinates 32.123, 34.456"),
    AIMessage(EntityExtraction(
        entity_type=EntityType.TARGET,
        explanation="The user explicitly mentions 'target' and provides coordinates, which is typical for target entities"
    ).model_dump_json()),
    
    HumanMessage("Send an alert to all users about system maintenance"),
    AIMessage(EntityExtraction(
        entity_type=EntityType.MESSAGE,
        explanation="The user wants to send an alert message to users, which matches the Message entity type"
    ).model_dump_json()),
    
    HumanMessage("Draw an arrow from point A to point B"),
    AIMessage(EntityExtraction(
        entity_type=EntityType.ARROW,
        explanation="The user wants to create a directional arrow between two points"
    ).model_dump_json())
]


class EntityNameExtractor:
    """Class for extracting entity names from queries.
    
    This class handles the extraction of entity types from various raw queries
    in the state object.
    
    Attributes
    ----------
    configuration : Configuration
        The configuration object for the agent.
    model : Any
        The chat model used for entity extraction.
    """
    
    def __init__(self):
        """Initialize the EntityNameExtractor with configuration and model."""
        self.configuration = Configuration.from_context()
        self.model = load_chat_model(self.configuration.model)
    
    async def _extract_entity_type(self, query: str) -> str:
        """Extract entity type from a single query.
        
        Parameters
        ----------
        query : str
            The input text to analyze.
            
        Returns
        -------
        str
            The extracted entity type.
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(SYSTEM_PROMPT),
            *FEW_SHOT,
            HumanMessagePromptTemplate.from_template("{input}"),
        ])

        splitter_chain = prompt | self.model | entity_parser
        result = await splitter_chain.ainvoke({"input": query})
        
        return result.entity_type.value
    
    async def extract_all_entities(self, state: State) -> Dict[str, str]:
        """Extract entity types from all relevant raw queries in the state.
        
        Parameters
        ----------
        state : State
            The current state object containing raw queries.
            
        Returns
        -------
        Dict[str, str]
            Dictionary mapping query types to their extracted entity types.
        """
        result = {}
        
        # Process query for edit
        if state.raw_query_params_for_edit:
            result["entity_type_action"] = await self._extract_entity_type(state.raw_query_params_for_edit)
            
        # Process source query
        if state.raw_source_query:
            result["entity_type_source"] = await self._extract_entity_type(state.raw_source_query)
            
        # Process destination query
        if state.raw_destination_query:
            result["entity_type_destination"] = await self._extract_entity_type(state.raw_destination_query)
            
        # Process trigger query if it exists
        if state.trigger_parts and state.is_geographic == False and state.trigger_parts.query:
            result["entity_type_query"] = await self._extract_entity_type(state.trigger_parts.query)
        
        if state.trigger_parts and state.trigger_parts.action and not state.query_params_for_edit:
            result["entity_type_action"] = await self._extract_entity_type(state.trigger_parts.action)
            
        # If no queries were processed, use the user's input
        if not result and state.input:
            if state.flow_type == "action":
                result["entity_type_action"] = await self._extract_entity_type(state.input)
            else:
                result["entity_type_query"] = await self._extract_entity_type(state.input)
            
        return result


async def entity_name_extract(state: State) -> Dict[str, str]:
    """Extract entity types from all raw queries in the state.
    
    Parameters
    ----------
    state : State
        The current state object containing raw queries.
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping query types to their extracted entity types.
    """
    extractor = EntityNameExtractor()
    return await extractor.extract_all_entities(state)