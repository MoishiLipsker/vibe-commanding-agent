"""API agent node."""

from typing import Dict, Any, List
import aiohttp
from fastapi.encoders import jsonable_encoder
from react_agent.state import State
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from react_agent.nodes.trigger_notification import TriggerNotificationHandler
from react_agent.configuration import Configuration
from react_agent.utils import load_chat_model
import random

class TriggerExplanationHandler:
    """Class for generating human-readable explanations of trigger creation.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    model : Any
        The chat model used for explanation generation
    """
    
    def __init__(self):
        """Initialize the TriggerExplanationHandler with model."""
        configuration = Configuration.from_context()
        self.model = load_chat_model(configuration.model)
    
    async def explain_trigger_creation(self, trigger_data: Dict[str, Any], trigger_response: Dict[str, Any]) -> str:
        """Generate a human-readable explanation of the trigger creation.
        
        Parameters
        ----------
        trigger_data : Dict[str, Any]
            The trigger configuration data
        trigger_response : Dict[str, Any]
            The response from the trigger creation API
            
        Returns
        -------
        str
            A human-readable explanation of what was created
        """
        system_prompt = """You are an assistant that generates concise notifications for trigger creation events.
        Provide a single, direct sentence in English, stating precisely what trigger was created and its main purpose.
        This notification is for a professional C&C system operator and should be brief and to the point.
        Include a brief instruction for managing the trigger on the triggers page."""
        
        trigger_info = f"""
        Trigger Configuration:
        Type: {trigger_data.get('type')}
        Source Query: {trigger_data.get('sourceQuery', trigger_data.get('query', {}))}
        Target Query: {trigger_data.get('targetQuery', {})}
        Raw Trigger: {trigger_data.get('rawTrigger')}
        Raw Action: {trigger_data.get('rawAction')}
        Actions: {trigger_data.get('actions')}
        
        API Response:
        {trigger_response}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=trigger_info)
        ]
        
        response = await self.model.ainvoke(messages)
        return response.content


class QueryExplanationHandler:
    """Class for generating human-readable explanations of query results.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    model : Any
        The chat model used for explanation generation
    """
    
    def __init__(self):
        """Initialize the QueryExplanationHandler with model."""
        configuration = Configuration.from_context()
        self.model = load_chat_model(configuration.model)
    
    async def explain_query_results(self, query_params: Dict[str, Any], query_response: List[Dict[str, Any]]) -> str:
        """Generate a human-readable explanation of the query results with markdown table.
        
        Parameters
        ----------
        query_params : Dict[str, Any]
            The query parameters used
        query_response : List[Dict[str, Any]]
            The response data from the query API
            
        Returns
        -------
        str
            A human-readable explanation with markdown table of results
        """
        system_prompt = """You are an assistant that generates concise notifications for query results.
        Provide a single, direct sentence in English, stating that the query was executed and briefly what it was for.
        Then, present a concise markdown table with the most relevant fields.
        This notification is for a professional C&C system operator.
        
        For the markdown table:
        - Include only fields meaningful to the end-user.
        - Exclude technical fields (e.g., IDs, timestamps) unless specifically relevant.
        - Use user-friendly table headers.
        - Right-align numeric columns.
        - Ensure correct markdown syntax."""
        
        query_info = f"""
        Query Parameters:
        {query_params}
        
        Query Results:
        {query_response}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query_info)
        ]
        
        response = await self.model.ainvoke(messages)
        return response.content


class EntityPreviewHandler:
    """Class for generating human-readable previews of entity creation.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    model : Any
        The chat model used for preview generation
    """
    
    def __init__(self):
        """Initialize the EntityPreviewHandler with model."""
        configuration = Configuration.from_context()
        self.model = load_chat_model(configuration.model)
    
    async def create_entity_preview(self, entity_data: Dict[str, Any]) -> str:
        """Generate a human-readable preview of the entity to be created with markdown table.
        
        Parameters
        ----------
        entity_data : Dict[str, Any]
            The entity data to be created
            
        Returns
        -------
        str
            A human-readable preview with markdown table and confirmation request
        """
        system_prompt = """You are an assistant that generates concise notifications for entity creation previews.
        Provide a single, direct sentence in English, stating that an entity is ready for creation.
        Then, present a concise markdown table with the entity details.
        This notification is for a professional C&C system operator.
        
        For the markdown table:
        - Include all essential fields defining the entity.
        - Field names should be user-friendly.
        - Values (dates, numbers, etc.) should be clearly formatted.
        - Ensure correct markdown syntax and appropriate alignment.
        
        Conclude with the exact phrase:
        'Proceed with creating this entity? (Respond \'yes\' to confirm, \'no\' to cancel).'"""
        
        entity_info = f"""
        Entity to be created:
        {entity_data}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=entity_info)
        ]
        
        response = await self.model.ainvoke(messages)
        return response.content


class UpdateExplanationHandler:
    """Class for generating human-readable explanations of entity updates.

    Parameters
    ----------
    None

    Attributes
    ----------
    model : Any
        The chat model used for explanation generation
    """

    def __init__(self):
        """Initialize the UpdateExplanationHandler with model."""
        configuration = Configuration.from_context()
        self.model = load_chat_model(configuration.model)

    async def explain_entity_update(self, action_data: Dict[str, Any], update_response: Dict[str, Any]) -> str:
        """Generate a human-readable explanation of the entity update.

        Parameters
        ----------
        action_data : Dict[str, Any]
            The data used for the update action, including query and payload
        update_response : Dict[str, Any]
            The response from the update API

        Returns
        -------
        str
            A human-readable explanation of what was updated
        """
        system_prompt = """You are an assistant that generates concise notifications for entity update events.
        Provide a single, direct sentence in English, stating precisely which entities were updated (or targeted for update), what was changed, and the outcome.
        This notification is for a professional C&C system operator and should be brief and to the point."""

        update_info = f"""
        Update Action Data:
        Target Query: {action_data.get('query')}
        Updates Applied: {action_data.get('updates')}

        API Response:
        {update_response}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=update_info)
        ]

        response = await self.model.ainvoke(messages)
        return response.content


async def api_agent(state: State) -> Dict[str, str]:
    """Execute API requests based on the current state.
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        Dict[str, str]: The response from the API.
    """
    base_url = "http://10.2.3.9:5000"
    explanation_handler = TriggerExplanationHandler()
    query_handler = QueryExplanationHandler()
    entity_preview_handler = EntityPreviewHandler()
    update_explanation_handler = UpdateExplanationHandler()
    
    try:
        # Handle geographic rule with entity update/create
        if state.source_query_params and state.destination_query_params:
            # Create the trigger with the updated format
            sourceQuery = {f"fields.{k}" if k != "type" else k: v for k, v in state.source_query_params.filters.items()}
            sourceQuery["type"] = state.source_query_params.entity_type
            targetQuery = {f"fields.{k}" if k != "type" else k: v for k, v in state.destination_query_params.filters.items()}
            targetQuery["type"] = state.destination_query_params.entity_type
            trigger_data = {
                "type": "geoRule",
                "sourceQuery": sourceQuery,
                "targetQuery": targetQuery,
                "rawTrigger": state.trigger_parts.query,
                "rawAction": state.trigger_parts.action,
                "actions": [{
                    "type": "updateEntity" if state.query_params_for_edit else "addEntity",
                    "payload": state.entity,
                    "query": state.query_params_for_edit if state.query_params_for_edit else {}
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                json_trigger_data = jsonable_encoder(trigger_data)
                print(json_trigger_data)
                async with session.post(f"{base_url}/triggers/new", json=json_trigger_data) as response:
                    trigger_response = await response.json()
                    explanation = await explanation_handler.explain_trigger_creation(trigger_data, trigger_response)
                    return {"response": explanation}
        
        # Handle regular query-based trigger
        elif state.trigger_parts:
            query_params = {f"fields.{k}" if k != "type" else k: v for k, v in state.query_params.filters.items()}
            query_params["type"] = state.query_params.entity_type
            trigger_data = {
                "type": "layer",
                "query": query_params,
                "rawTrigger": state.trigger_parts.query,
                "rawAction": state.trigger_parts.action,
                "actions": [{
                    "type": "updateEntity" if state.query_params_for_edit else "createEntity",
                    "payload": state.entity,
                    "query": state.query_params_for_edit if state.query_params_for_edit else {}
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                json_trigger_data = jsonable_encoder(trigger_data)
                print(json_trigger_data)
                async with session.post(f"{base_url}/triggers/new", json=json_trigger_data) as response:
                    trigger_response = await response.json()
                    explanation = await explanation_handler.explain_trigger_creation(trigger_data, trigger_response)
                    return {"response": explanation}
        
        # Handle simple entity update
        elif state.query_params_for_edit:
            query_params = {f"fields.{k}" if k != "type" else k: v for k, v in state.query_params_for_edit.filters.items()}  
            query_params["type"] = state.query_params_for_edit.entity_type
            action_data = {
                "updates": state.entity,
                "query": query_params
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/entities/update-by-query/",
                    json=jsonable_encoder(action_data)
                ) as response:
                    update_response = await response.json()
                    explanation = await update_explanation_handler.explain_entity_update(action_data, update_response)
                    return {"response": explanation}
        
        # Handle simple entity creation
        elif state.entity:
            entity = state.entity
            entity.fields["position"] = entity.position
            if "name" not in entity.fields or entity.fields["name"] is None:
                # add generic name if not provided
                entity.fields["name"] = entity.type + " " + str(random.randint(1, 1000000))
                

            async with aiohttp.ClientSession() as session:
                 async with session.post(f"{base_url}/entities/add", json=jsonable_encoder(entity)) as response:
                     create_response = await response.json()
                     print(create_response)
                     #return {"response": f"Successfully created entity: {create_response}"}
            # First, generate a preview and ask for confirmation
            preview = await entity_preview_handler.create_entity_preview(entity)
            return {
                "response": preview,
                "requires_confirmation": True,
                "confirmation_data": {
                    "action": "create_entity",
                    "entity": entity
                }
            }
        
        # Handle queries
        elif state.query_params:
            query_params = {f"fields.{k}" if k != "type" else k: v for k, v in state.query_params.filters.items()}
            query_params["type"] = state.query_params.entity_type
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/entities/query",
                    json=jsonable_encoder(query_params)
                ) as response:
                    query_response = await response.json()
                    print(response.url)
                    explanation = await query_handler.explain_query_results(query_params, query_response)
                    return {"response": explanation}
        
        return {"response": "No valid operation found in state"}
        
    except Exception as e:
        return {"response": f"Error: {str(e)}"}


async def handle_entity_creation_confirmation(entity: Dict[str, Any], base_url: str = "http://10.2.3.9:5000") -> Dict[str, str]:
    """Handle the actual entity creation after confirmation.
    
    Args:
        entity (Dict[str, Any]): The entity data to create
        base_url (str): The base URL for the API
        
    Returns:
        Dict[str, str]: The response from the API
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{base_url}/entities/add", json=jsonable_encoder(entity)) as response:
            create_response = await response.json()
            return {"response": f"Successfully created entity: {create_response}"} 