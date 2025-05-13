"""Define the state management for the React Agent."""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Sequence
from pydantic import BaseModel, Field
from enum import Enum
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages



class ActionType(str, Enum):
    """Types of actions that can be performed."""
    CREATE = "create_entity"
    UPDATE = "update_entity"
    # UPDATE_WITHOUT_QUERY = "update_entity_without_query"

# --- Type Definitions ---


class QueryParams(BaseModel):
    entity_type: str
    filters: Dict[str, Any]

class TriggerParts(BaseModel):
    """Represents a trigger object with query and action"""
    query: str = Field(..., description="Natural language query for the trigger")
    action: str = Field(..., description="Natural language action for the trigger")

class ProcessedTrigger(BaseModel):
    query: QueryParams
    action: EntityData

class EntityData(BaseModel):
    type: str = Field(..., description="The type of entity being created")
    properties: Dict = Field(..., description="The entity properties following the schema")

class InputState(BaseModel):
    """Input state for the workflow."""
    #messages: List[Dict[str, str]]
    input: Optional[str] = None
    #entities: Optional[Dict] = None
    #schemas: Optional[Dict] = None

class OutputState(BaseModel):
    """output state for the workflow."""
    response: Optional[str] = Field(None, description="Response from the API call")
    messages: Annotated[Sequence[BaseMessage], add_messages]

class State(BaseModel):
    """State for the workflow."""
    input: str = Field(..., description="The user's input")
    messages: Annotated[Sequence[BaseMessage], add_messages]
    flow_type: Optional[str] = Field(None, description="The type of flow: trigger, query, or action")

    trigger_parts: Optional[TriggerParts] = Field(None, description="The parts of a trigger")
    is_geographic: Optional[bool] = Field(None, description="Whether the query is geographic")

    entity_type_query: Optional[str] = Field(None, description="The type of entity to query")
    entity_type_action: Optional[str] = Field(None, description="The type of entity to action")
    entity_type_source: Optional[str] = Field(None, description="The type of entity to source")
    entity_type_destination: Optional[str] = Field(None, description="The type of entity to destination")

    raw_query_params_for_edit: Optional[str] = Field(None, description="Query parameters for edit operations")
    raw_source_query: Optional[str] = Field(None, description="The raw source")
    raw_destination_query: Optional[str] = Field(None, description="The raw destination")

    query_params: Optional[QueryParams] = Field(None, description="The query parameters")
    query_params_for_edit: Optional[QueryParams] = Field(None, description="The query parameters for edit operations")
    source_query_params: Optional[QueryParams] = Field(None, description="The source part of a geographic query")
    destination_query_params: Optional[QueryParams] = Field(None, description="The destination part of a geographic query")

    action_type: Optional[ActionType] = Field(None, description="The type of action")
    entity: Optional[EntityData] = Field(None, description="The action entity parameters")
       
    # API specific
    api_response: Optional[dict] = Field(None, description="Response from API calls")
    user_response: Optional[str] = Field(None, description="User response to the API call")
    response: Optional[str] = Field(None, description="Response from the API call")
    
    # Status tracking
    status: Optional[str] = Field(None, description="Current status of the workflow")
    error: Optional[str] = Field(None, description="Error message if any")
    errors: Annotated[Sequence[str], add_messages] = Field(default_factory=list, description="List of errors that occurred during processing")
    class Config:
        arbitrary_types_allowed = True
