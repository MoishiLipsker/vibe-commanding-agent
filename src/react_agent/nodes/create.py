"""Create flow node."""

from typing import Dict

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import BaseModel, Field

from react_agent.configuration import Configuration
from react_agent.state import State, EntityData
from react_agent.utils import load_chat_model


entity_parser = PydanticOutputParser(pydantic_object=EntityData)


def generate_few_shot_examples(entity_type: str, schema: str) -> list:
    """Generate few-shot examples based on the entity schema.
    
    Args:
        entity_type (str): The type of entity being created
        schema (str): The entity schema
        
    Returns:
        list: List of few-shot examples
    """
    # This is a simplified example - in a real implementation, you would want to
    # parse the schema and generate more sophisticated examples
    return [
        HumanMessage(f"Create a new {entity_type} with all required fields"),
        AIMessage(EntityData(
            type=entity_type,
            data={
                "example_field": "example_value",
                "required_field": "required_value"
            }
        ).model_dump_json())
    ]


async def action_parser(state: State) -> Dict[str, EntityData]:
    """Process the action part into structured entity data.
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        Dict[str, Entity]: Structured entity data.
    """
     # First, use entity_name_extract to identify the entity type
    entity_type = state.entity_type_action
    if entity_type is None:
        return {}
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model)
    
    # Use the appropriate action text based on the flow
    action_text = state.trigger_parts.action if state.trigger_parts else state.input
    
    # Get the entity schema
    schema = await configuration.get_entity_schema(entity_type)
    
    SYSTEM_PROMPT = f"""
    Your task is to create ONLY the fields that the user explicitly requested in their entity creation request.
    Do not try to create a complete entity - other fields will be added in a different process.
    
    Important guidelines:
    1. Extract ONLY the fields that the user explicitly mentioned or implied in their request
    2. Use ONLY fields that exist in the schema
    3. If a field in the request doesn't exist in the schema, find the closest matching field
    4. If multiple fields could match, choose the most appropriate one
    5. Never say you cannot create the entity - always create the closest possible match for the requested fields
    6. Do not add any fields that weren't requested by the user
    7. Do not worry about required fields - they will be handled separately

    Example (for illustration only - use the actual schema below):
    If user says: "Create a target with high priority at coordinates 32.123, 34.456"
    And the schema has fields: priority, location
    You should return ONLY:
    {{
        "type": "target",
        "properties": {{
            "priority": "high"
        }},
        "position": {{
            "lat": 32.123,
            "lng": 34.456
        }}
    }}
    Note: This is just an example. The actual schema and fields will be different.
    
    Entity Schema:
    {schema}

    Entity position (if not provided by the user):
    {state.position}
    
    Expected output JSON structure:
    {entity_parser.get_format_instructions()}

    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])

    splitter_chain = prompt | model | entity_parser

    result = await splitter_chain.ainvoke({"input": action_text})
    
    return {"entity": result} 