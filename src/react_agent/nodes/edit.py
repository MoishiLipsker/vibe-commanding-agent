"""Edit flow node."""

from typing import Dict
from enum import Enum

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import BaseModel, Field

from react_agent.configuration import Configuration
from react_agent.state import State
from react_agent.utils import load_chat_model


class QueryType(str, Enum):
    """Available query types for entity search."""
    BY_NAME = "by_name"
    BY_ID = "by_id"
    BY_EMAIL = "by_email"
    BY_TITLE = "by_title"
    BY_CUSTOM = "by_custom"


class QueryExtractionResult(BaseModel):
    """Query extraction result."""
    query_text: str = Field(..., description="The extracted natural language query")
    query_type: QueryType = Field(..., description="Type of query being performed")
    explanation: str = Field(..., description="Explanation of how the query was extracted")
    original_language: str = Field(..., description="The original language of the query")


query_parser = PydanticOutputParser(pydantic_object=QueryExtractionResult)

SYSTEM_PROMPT = """
You are a precise query extraction assistant that converts action text into natural language queries to find entities.
You can process queries in any language and maintain the original language in the response.

Key Guidelines:
1. Extract ONLY the search/find portion of the action text
2. Maintain all specific identifiers (names, emails, IDs, etc.)
3. Convert action verbs (update, change, modify, set) to search verbs (find, locate, get)
4. Preserve exact quotes and special characters
5. Remove any modification-related text
6. Keep the query in the original language of the input
7. Detect and note the original language of the query

Example Transformations:
- "update the goal named 'Complete project documentation' to high priority" 
  → "find the goal named 'Complete project documentation'"
- "change the user with email 'john.doe@example.com' display name" 
  → "find the user with email 'john.doe@example.com'"
- "modify the task called 'Implement login page' status" 
  → "find the task called 'Implement login page'"
- "set the project titled 'Q4 Website Redesign' deadline" 
  → "find the project titled 'Q4 Website Redesign'"

Expected output JSON structure:
{format_instructions}
"""

FEW_SHOT = [
    HumanMessage("update the goal named 'Complete project documentation' to high priority"),
    AIMessage(QueryExtractionResult(
        query_text="find the goal named 'Complete project documentation'",
        query_type=QueryType.BY_NAME,
        explanation="Extracted name-based query from update action, preserving exact goal name",
        original_language="en"
    ).model_dump_json()),
    
    HumanMessage("change the user with email 'john.doe@example.com' display name"),
    AIMessage(QueryExtractionResult(
        query_text="find the user with email 'john.doe@example.com'",
        query_type=QueryType.BY_EMAIL,
        explanation="Extracted email-based query from change action, preserving exact email address",
        original_language="en"
    ).model_dump_json()),

    HumanMessage("modify the task with custom field 'project_id' equals 'PRJ-123' status"),
    AIMessage(QueryExtractionResult(
        query_text="find the task with custom field 'project_id' equals 'PRJ-123'",
        query_type=QueryType.BY_CUSTOM,
        explanation="Extracted custom field-based query from modify action, preserving exact field name and value",
        original_language="en"
    ).model_dump_json()),

    HumanMessage("עדכן את המשימה בשם 'השלמת תיעוד הפרויקט' לסטטוס גבוה"),
    AIMessage(QueryExtractionResult(
        query_text="מצא את המשימה בשם 'השלמת תיעוד הפרויקט'",
        query_type=QueryType.BY_NAME,
        explanation="Extracted name-based query from update action in Hebrew, preserving exact task name",
        original_language="he"
    ).model_dump_json())
]


async def action_extract(state: State) -> Dict[str, str]:
    """Process the edit part into structured query parameters.
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        Dict[str, str]: Structured query parameters.
    """
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model)
    
    # Use the appropriate action text based on the flow
    action_text = state.trigger_parts.action if state.trigger_parts else state.input

    # Format the system prompt with parser instructions
    formatted_system_prompt = SYSTEM_PROMPT.format(
        format_instructions=query_parser.get_format_instructions()
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(formatted_system_prompt),
        *FEW_SHOT,
        HumanMessagePromptTemplate.from_template("{input}"),
    ])

    chain = prompt | model | query_parser

    result = await chain.ainvoke({"input": action_text})
    
    return {"raw_query_params_for_edit": result.query_text}