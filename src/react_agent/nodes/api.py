"""API agent node."""

from typing import Dict, Any
import aiohttp
from fastapi.encoders import jsonable_encoder
from react_agent.state import State
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

def to_dict(obj: Any) -> Dict[str, Any]:
    """Convert an object to a dictionary.
    
    Args:
        obj: The object to convert
        
    Returns:
        Dict[str, Any]: The dictionary representation of the object
    """
    if hasattr(obj, '__dict__'):
        return {k: to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_dict(item) for item in obj]
    else:
        return obj

async def api_agent(state: State) -> Dict[str, str]:
    """Execute API requests based on the current state.
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        Dict[str, str]: The response from the API.
    """
    base_url = "http://10.2.3.9:5000"
    
    try:
        # Handle geographic rule with entity update/create
        if state.source_query_params and state.destination_query_params:
            # Create the trigger with the updated format
            trigger_data = {
                "type": "geoRule",
                "sourceQuery": {
                    "properties": state.source_query_params
                },
                "targetQuery": state.destination_query_params,
                "actions": [{
                    "type": "updateEntity" if state.query_params_for_edit else "createEntity",
                    "payload": {
                        "properties": state.entity
                    },
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
            trigger_data = {
                "type": "queryRule",
                "sourceQuery": state.query_params,
                "actions": [{
                    "type": "updateEntity" if state.query_params_for_edit else "createEntity",
                    "payload": {
                        "properties": state.entity
                    },
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
                "type": "updateEntity",
                "payload": {
                    "properties": state.entity
                },
                "query": state.query_params_for_edit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    f"{base_url}/entities/update/",
                    json=jsonable_encoder(action_data)
                ) as response:
                    update_response = await response.json()
                    return {"response": f"Successfully updated entity: {update_response}"}
        
        # Handle simple entity creation
        elif state.entity:                       
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{base_url}/entities/add", json=jsonable_encoder(state.entity)) as response:
                    create_response = await response.json()
                    return {"response": f"Successfully created entity: {create_response}"}
        
        # Handle queries
        elif state.query_params:       
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{base_url}/entities/query",
                    params=jsonable_encoder(state.query_params)
                ) as response:
                    query_response = await response.json()
                    return {"response": f"Query results: {query_response}"}
        
        return {"response": "No valid operation found in state"}
        
    except Exception as e:
        return {"response": f"Error: {str(e)}"} 