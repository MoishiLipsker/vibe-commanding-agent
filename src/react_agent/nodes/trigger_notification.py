"""Trigger notification handler using LLM."""

from typing import Dict, Any
from pydantic import BaseModel
from langchain_core.messages import AIMessage, SystemMessage
from react_agent.configuration import Configuration
from react_agent.utils import load_chat_model


class TriggerNotificationHandler:
    """Class for handling trigger notifications using LLM.
    
    Parameters
    ----------
    model_config : Configuration
        The configuration for the LLM model.
        
    Attributes
    ----------
    model : Any
        The loaded LLM model.
    """
    
    def __init__(self, model_config: Configuration):
        """Initialize the trigger notification handler.
        
        Parameters
        ----------
        model_config : Configuration
            The configuration for the LLM model.
        """
        self.model = load_chat_model(model_config.model)
    
    async def format_trigger_event(self, trigger_data: Dict[str, Any]) -> str:
        """Format trigger event data into a human-readable message using LLM.
        
        Parameters
        ----------
        trigger_data : Dict[str, Any]
            The trigger event data containing information about what happened.
            
        Returns
        -------
        str
            A human-readable message explaining what happened.
        """
        # Create system prompt to guide the LLM
        system_prompt = """You are a notification system that explains system events to users.
        When a trigger event occurs, explain clearly and concisely:
        1. What event occurred (based on event_data)
        2. Which trigger was activated by this event (based on trigger_data)
        3. What actions were performed by the system as a result

        Important:
        - Use clear and simple language
        - Keep explanations brief and relevant
        - Focus on the key details that users need to understand
        - Format the response as a short paragraph, not a list
        """
        
        # Create prompt for specific event
        event_prompt = f"""I received the following event:

        Activated Trigger Details:
        {trigger_data['trigger_data']}
        
        Event Details:
        {trigger_data['event_data']}
        
        Please explain clearly what happened in the system."""
        
        # Get LLM response
        messages = [
            SystemMessage(content=system_prompt),
            AIMessage(content=event_prompt)
        ]
        response = await self.model.ainvoke(messages)
        
        return response.content

    async def process_trigger_event(self, trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a trigger event and generate a notification message.
        
        Parameters
        ----------
        trigger_data : Dict[str, Any]
            The trigger event data.
            
        Returns
        -------
        Dict[str, Any]
            Response containing the formatted message.
        """
        formatted_message = await self.format_trigger_event(trigger_data)
        
        return {
            "response": formatted_message,
            "messages": [AIMessage(content=formatted_message)]
        } 