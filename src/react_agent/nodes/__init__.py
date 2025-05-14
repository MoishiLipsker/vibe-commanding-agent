"""Node functions for the workflow."""

from functools import wraps
from typing import Dict, Any, Callable
from react_agent.state import State

def error_handler(func: Callable) -> Callable:
    """Wrap a node function with error handling.
    
    Parameters
    ----------
    func : Callable
        The node function to wrap
        
    Returns
    -------
    Callable
        The wrapped function with error handling
    """
    @wraps(func)
    async def wrapper(state: State, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        try:
            return await func(state, *args, **kwargs)
        except Exception as e:
            return {"errors": [str(e)]}
    return wrapper

from .classify import classify_input
from .trigger import trigger_extract
from .query import query_parser
from .action import check_action
from .create import action_parser
from .edit import action_extract
from .api import api_agent
from .entity_name import entity_name_extract
from .geographic import geographic_extract
from .check_geographic import check_geographic

# Wrap all node functions with error handling only
classify_input = error_handler(classify_input)
trigger_extract = error_handler(trigger_extract)
check_geographic = error_handler(check_geographic)
geographic_extract = error_handler(geographic_extract)
check_action = error_handler(check_action)
action_extract = error_handler(action_extract)
query_parser = error_handler(query_parser)
action_parser = error_handler(action_parser)
api_agent = error_handler(api_agent)
entity_name_extract = error_handler(entity_name_extract)

__all__ = [
    "classify_input",
    "trigger_extract",
    "check_geographic",
    "geographic_extract",
    "check_action",
    "action_extract",
    "query_parser",
    "action_parser",
    "api_agent",
    "entity_name_extract",
] 