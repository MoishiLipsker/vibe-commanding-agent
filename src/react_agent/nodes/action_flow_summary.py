"""Node for providing natural language explanations of action flow events and triggers."""

from typing import Dict, AsyncGenerator, Any, List
from dataclasses import dataclass
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from react_agent.configuration import Configuration
from react_agent.state import State
from react_agent.utils import load_chat_model
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


@dataclass
class ActionFlowEvent:
    """
    Represents an action flow event with its associated data.

    Parameters
    ----------
    trigger_query : str
        The trigger query that initiated the flow
    entity_data : Dict[str, Any]
        The entity data received with the trigger
    planned_action : str
        The action that will be executed
    """
    trigger_query: str
    entity_data: List[Dict[str, Any]]
    planned_action: str


class ActionFlowSummaryHandler:
    """
    Handles explanation generation for action flow events using LLM.
    
    This class processes trigger events and actions in the action flow context,
    generating clear explanations of what occurred and what will happen next.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    configuration : Configuration
        The configuration object for the agent
    model : Any
        The chat model used for explanation generation
    """
    
    def __init__(self):
        """Initialize the ActionFlowSummaryHandler with configuration and model."""
        self.configuration = Configuration.from_context()
        self.model = load_chat_model(self.configuration.model)
        
    async def explain_event(self, event: ActionFlowEvent) -> AsyncGenerator[Dict[str, str], None]:
        """
        Generate a streaming explanation of an action flow event.
        
        Parameters
        ----------
        event : ActionFlowEvent
            The event to explain containing trigger, entity and action data
            
        Yields
        ------
        Dict[str, str]
            Dictionary containing the streaming explanation message
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an assistant that generates concise notifications for action flow events.
            Provide a single, direct sentence in English, stating precisely what action will be taken and its direct trigger.
            This notification is for a professional C&C system operator and should be brief and to the point."""),
            HumanMessagePromptTemplate.from_template(
                """Action flow event details:
Trigger: {trigger}
Entities: {entities}
Planned Action: {action}
Notification:"""
            )
        ])
        
        chain = prompt | self.model | StrOutputParser()
        full_explanation = ""
        
        async for chunk in chain.astream({
            "trigger": event.trigger_query,
            "entities": event.entity_data,
            "action": event.planned_action
        }):
            full_explanation += chunk
            yield {"explanation": full_explanation}


async def process_action_flow_event(
    trigger_query: str,
    entities: List[Dict[str, Any]],
    planned_action: str
) -> str:
    """
    Generate a streaming explanation of an action flow event.
    
    Parameters
    ----------
    trigger_query : str
        The trigger query that initiated the flow
    entity_data : Dict[str, Any]
        The entity data received with the trigger
    planned_action : str
        The action that will be executed
        
    Returns
    -------
    str
        A string containing the full explanation message
    """
    handler = ActionFlowSummaryHandler()
    event = ActionFlowEvent(
        trigger_query=trigger_query,
        entities=entities,
        planned_action=planned_action
    )
    
    full_explanation = ""
    async for explanation_chunk in handler.explain_event(event):
        full_explanation = explanation_chunk["explanation"]
    return full_explanation 