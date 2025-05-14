"""LLM Judge wrapper for validating node responses.

This module provides a wrapper class that uses an LLM to validate if a node's response
is logical and appropriate for the given prompt.
"""

from functools import wraps
from typing import Any, Callable, Dict, Optional, Union
import openai
from react_agent.state import State


class JudgeWrapper:
    """A wrapper class that validates node responses using an LLM judge.
    
    Parameters
    ----------
    max_retries : int, optional
        Maximum number of retries if validation fails, by default 3
    
    Attributes
    ----------
    max_retries : int
        Maximum number of retries allowed
    """
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
    
    async def validate_response(self, prompt: Union[str, Dict[str, Any]], response: Dict[str, Any], node_name: str) -> bool:
        """Validate if the response is logical for the given prompt using LLM.
        
        Parameters
        ----------
        prompt : Union[str, Dict[str, Any]]
            The original prompt or messages sent to the LLM in the node
        response : Dict[str, Any]
            The node's response to validate
        node_name : str
            The name of the node being validated
            
        Returns
        -------
        bool
            True if response is valid, False otherwise
        """
        try:
            # Format the prompt based on its type
            if isinstance(prompt, dict):
                formatted_prompt = f"Messages: {prompt}"
            else:
                formatted_prompt = f"Prompt: {prompt}"
            
            # Construct the validation prompt
            validation_prompt = f"""As an objective judge, evaluate if the following response from the '{node_name}' node is logical and appropriate 
            for the given prompt. Answer with just 'yes' or 'no'.
            
            {formatted_prompt}
            Response: {response}
            
            Consider:
            1. Does the response directly address what was asked in the prompt?
            2. Is the response format appropriate for this type of node?
            3. Does the response contain all necessary information?
            
            Is this response logical and appropriate?"""
            
            # Call OpenAI API
            completion = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an objective judge evaluating if responses are logical and appropriate for specific node operations."},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.1,  # Low temperature for more consistent judgments
                max_tokens=10
            )
            
            # Get the judgment
            judgment = completion.choices[0].message.content.strip().lower()
            return judgment == "yes"
            
        except Exception as e:
            print(f"Error in validation for {node_name}: {str(e)}")
            return True  # Default to True in case of errors to avoid blocking the flow
    
    def __call__(self, func: Callable) -> Callable:
        """Wrap a node function with LLM validation.
        
        Parameters
        ----------
        func : Callable
            The node function to wrap
            
        Returns
        -------
        Callable
            The wrapped function with validation
        """
        @wraps(func)
        async def wrapper(state: State, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            retries = 0
            
            # Get the node name from the function
            node_name = func.__name__
            
            while retries < self.max_retries:
                # Store the original prompt/messages before executing the function
                original_prompt = kwargs.get('prompt') or kwargs.get('messages') or getattr(state, 'prompt', None) or getattr(state, 'messages', None)
                if not original_prompt:
                    # If we can't find the prompt, proceed without validation
                    return await func(state, *args, **kwargs)
                
                # Execute the node function
                response = await func(state, *args, **kwargs)
                
                # Validate the response
                is_valid = await self.validate_response(original_prompt, response, node_name)
                
                if is_valid:
                    return response
                    
                retries += 1
                if retries < self.max_retries:
                    print(f"Response validation failed for {node_name}, attempt {retries + 1} of {self.max_retries}")
                
            # If we've exhausted retries, return the last response
            print(f"Maximum retries reached for {node_name}, proceeding with last response")
            return response
            
        return wrapper 