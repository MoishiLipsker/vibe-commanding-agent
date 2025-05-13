"""Query flow nodes."""

from typing import Dict, cast
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableConfig

from react_agent.configuration import Configuration
from react_agent.state import State, QueryParams
from react_agent.utils import load_chat_model

parser = PydanticOutputParser(pydantic_object=QueryParams)

class QueryParser:
    """Class for parsing queries into structured parameters.
    
    This class handles the generation of structured queries for different entity types
    in the state object.
    
    Attributes
    ----------
    configuration : Configuration
        The configuration object for the agent.
    model : Any
        The chat model used for query generation.
    """
    
    def __init__(self):
        """Initialize the QueryParser with configuration and model."""
        self.configuration = Configuration.from_context()
        self.model = load_chat_model(self.configuration.model)
        # self.model.with_structured_output(QueryParams)
    
    async def _generate_query_for_entity(self, entity_type: str, query_text: str) -> QueryParams:
        """Generate structured query parameters for a specific entity type.
        
        Parameters
        ----------
        entity_type : str
            The type of entity to generate query for.
        query_text : str
            The raw query text to process.
            
        Returns
        -------
        QueryParams
            Structured query parameters for the entity.
        """
        schema = await self.configuration.get_entity_schema(entity_type)
        
        SYSTEM_PROMPT = f"""
        Generate a structured query based on the user's request and the entity schema.
        
        Important guidelines:
        1. Use ONLY fields that exist in the schema
        2. If a field in the request doesn't exist in the schema, find the closest matching field
        3. If multiple fields could match, choose the most appropriate one
        4. Include only the fields that are relevant to the query
        
        The following examples are for illustration only - they may not match the actual schema fields.
        You must use ONLY the fields that exist in the schema provided below.
        
        Entity Type:
        {entity_type}
        
        Entity Schema:
        {schema}
        
        Expected output JSON structure:
        {parser.get_format_instructions()}
        """
        
        FEW_SHOT = [
            HumanMessage("targets with high priority in sector A"),
            AIMessage(QueryParams(
                entity_type="Target",
                filters={
                    "priority": "high",
                    "location": {"sector": "A"}
                },
            ).model_dump_json()),
            
            HumanMessage("friendly forces that are active"),
            AIMessage(QueryParams(
                entity_type="FriendlyForce",
                filters={
                    "status": "active",
                    "type": "friendly"
                },
            ).model_dump_json()),
            
            HumanMessage("ISR tasks assigned to unit Bravo"),
            AIMessage(QueryParams(
                entity_type="ISRTask",
                filters={
                    "assigned_to": "Bravo"
                },
            ).model_dump_json())
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(SYSTEM_PROMPT),
            *FEW_SHOT,
            HumanMessagePromptTemplate.from_template("return the query parameters for the following query: {input}"),
        ])
        
        chain = prompt | self.model | parser
        
        result = await chain.ainvoke({"input": query_text})
        return result
    
    async def parse_all_queries(self, state: State) -> Dict[str, QueryParams]:
        """Generate structured queries for all entity types in the state.
        
        Parameters
        ----------
        state : State
            The current state object containing entity types and queries.
            
        Returns
        -------
        Dict[str, QueryParams]
            Dictionary mapping query types to their structured parameters.
        """
        result = {}
        
        # Process source query
        if state.entity_type_source and state.raw_source_query:
            result["source_query_params"] = await self._generate_query_for_entity(
                state.entity_type_source,
                state.raw_source_query
            )
            
        # Process destination query
        if state.entity_type_destination and state.raw_destination_query:
            result["destination_query_params"] = await self._generate_query_for_entity(
                state.entity_type_destination,
                state.raw_destination_query
            )
            
        # Process action query
        if state.entity_type_action and state.raw_query_params_for_edit:
            result["query_params_for_edit"] = await self._generate_query_for_entity(
                state.entity_type_action,
                state.raw_query_params_for_edit
            )
            
        # Process trigger query if it exists
        if state.trigger_parts and state.trigger_parts.query and state.entity_type_query:
            result["query_params"] = await self._generate_query_for_entity(
                state.entity_type_query,
                state.trigger_parts.query
            )
            
        # If no queries were processed and we have user input, use that
        if not result and state.input and state.entity_type_query:
            result["query_params"] = await self._generate_query_for_entity(
                state.entity_type_query,
                state.input
            )
            
        return result


async def query_parser(state: State) -> Dict[str, QueryParams]:
    """Process all queries in the state into structured parameters.
    
    Parameters
    ----------
    state : State
        The current state of the conversation.
        
    Returns
    -------
    Dict[str, QueryParams]
        Dictionary containing structured query parameters for all entity types.
    """
    parser = QueryParser()
    return await parser.parse_all_queries(state) 