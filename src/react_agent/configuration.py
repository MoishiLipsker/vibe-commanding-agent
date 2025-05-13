"""Define the configurable parameters for the agent."""

from __future__ import annotations

from typing import Annotated, Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field
import json
import os
import aiofiles
import asyncio

class EntitySchema(BaseModel):
    """Schema definition for an entity type"""
    name: str = Field(..., description="Name of the entity type")
    fields: Dict[str, dict] = Field(..., description="Field definitions for the entity")
    description: str = Field(..., description="Description of the entity type")

class LLMConfig(BaseModel):
    """Configuration for LLM settings"""
    model: str = Field(default="anthropic/claude-3-sonnet-20240229", description="The LLM model to use")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: int = Field(default=1000, description="Maximum tokens for generation")

# Define OperationType as a Literal type
OperationType = Literal["create_entity", "query", "create_trigger", "auto"]

class Configuration(BaseModel):
    """Configuration for the entity management agent."""

    model: str = Field(
        default="openai/gpt-4o",
        description="The model to use for the agent.",
    )
    
    operation_type: OperationType = Field(
        default="auto",
        description="The type of operation to perform. If 'auto', the agent will determine the operation type from the user's request.",
    )
    
    max_steps: int = Field(
        default=5,
        description="Maximum number of steps the agent can take before giving up.",
    )
    
    recursion_limit: int = Field(
        default=2,
        description="Maximum number of recursive calls allowed.",
    )

    schema_path: str = Field(
        default="schemas",
        description="Path to the directory containing entity schemas"
    )

    @classmethod
    def from_context(cls) -> "Configuration":
        """Create a configuration from the current context."""
        return cls()

    async def get_entity_schema(self, entity_type: str) -> Dict:
        """Get the schema for a specific entity type asynchronously.
        
        Args:
            entity_type (str): The type of entity to get the schema for.
            
        Returns:
            Dict: The entity schema.
            
        Raises:
            ValueError: If the entity type is not found or schema file is invalid.
        """
        schema_file = os.path.join(self.schema_path, f"{entity_type.lower()}.json")
        
        if not os.path.exists(schema_file):
            raise ValueError(f"Schema file not found for entity type: {entity_type}")
        
        try:
            async with aiofiles.open(schema_file, 'r') as f:
                content = await f.read()
                schema = json.loads(content)
                
            # Validate schema structure
            if not isinstance(schema, dict):
                raise ValueError(f"Invalid schema format for {entity_type}")
            
            if "fields" not in schema:
                raise ValueError(f"Schema for {entity_type} missing 'fields' section")
            
            return schema
            
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in schema file for {entity_type}")
        except Exception as e:
            raise ValueError(f"Error loading schema for {entity_type}: {str(e)}")

    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "llm_config": {
                    "model": "anthropic/claude-3-sonnet-20240229",
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            }
        }
