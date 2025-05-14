"""API agent node."""

from typing import Dict, Any
import aiohttp
from fastapi.encoders import jsonable_encoder
from react_agent.state import State
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from react_agent.nodes.trigger_notification import TriggerNotificationHandler
from react_agent.configuration import Configuration

async def api_agent(state: State) -> Dict[str, str]:
    """Execute API requests based on the current state.
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        Dict[str, str]: The response from the API.
    """
    base_url = "http://10.2.3.9:5000"
    config = Configuration.from_context()
    notification_handler = TriggerNotificationHandler(config)
    
    try:
        # Handle geographic rule with entity update/create
        if state.source_query_params and state.destination_query_params:
            # Create the trigger with the updated format
            sourceQuery = state.source_query_params.filters
            sourceQuery["type"] = state.source_query_params.entity_type
            targetQuery = state.destination_query_params.filters
            targetQuery["type"] = state.destination_query_params.entity_type
            trigger_data = {
                "type": "geoRule",
                "sourceQuery": sourceQuery,
                "targetQuery": targetQuery,
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
                    return {"response": f"Successfully created trigger: {trigger_response}",
                            "messages": [AIMessage(f"Successfully created trigger: {trigger_response}")]}
        
        # Handle regular query-based trigger
        elif state.trigger_parts:
            query_params = state.query_params.filters
            query_params["type"] = state.query_params.entity_type
            trigger_data = {
                "type": "layer",
                "query": query_params,
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
                    return {"response": f"Successfully created trigger: {trigger_response}",
                            "messages": [AIMessage(f"Successfully created trigger: {trigger_response}")]}
        
        # Handle simple entity update
        elif state.query_params_for_edit:
            action_data = {
                "updates": state.entity,
                "query": state.query_params_for_edit.filters #add type
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/entities/update-by-query/",
                    json=jsonable_encoder(action_data)
                ) as response:
                    update_response = await response.json()
                    return {"response": f"Successfully updated entity: {update_response}"}
        
        # Handle simple entity creation
        elif state.entity:                       
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{base_url}/entities/add", json=jsonable_encoder(state.entity)) as response:
                    create_response = await response.json()
                    return {"response": f"Successfully created entity: {create_response}",
                            "messages": [AIMessage(f"Successfully created entity: {create_response}")]}
        
        # Handle queries
        elif state.query_params:
            query_params = state.query_params.filters
            query_params["type"] = state.query_params.entity_type
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/entities/query",
                    json=jsonable_encoder(query_params)
                ) as response:
                    query_response = await response.json()
                    print(response.url)
                    return {"response": f"Query results: {query_response}",
                            "messages": [AIMessage(f"Query results: {query_response}")]}
        
        return {"response": "No valid operation found in state"}
        
    except Exception as e:
        return {"response": f"Error: {str(e)}"} 