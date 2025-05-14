"""Process summary node that provides a natural language description of what happened."""

from typing import Dict, AsyncGenerator
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, AIMessageChunk
from langchain_core.output_parsers import StrOutputParser
from react_agent.configuration import Configuration
from react_agent.state import State
from react_agent.utils import load_chat_model
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

class ProcessSummaryHandler:
    """Class for generating natural language summaries of process execution.
    
    This class handles the generation of human-readable summaries of what happened
    during the request processing, using the LLM to create a clear explanation.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    configuration : Configuration
        The configuration object for the agent
    model : Any
        The chat model used for summary generation
    """
    
    def __init__(self):
        """Initialize the ProcessSummaryHandler with configuration and model."""
        self.configuration = Configuration.from_context()
        self.model = load_chat_model(self.configuration.model)
        
    async def generate_summary(self, state: State) -> AsyncGenerator[Dict[str, str], None]:
        """Generate a streaming summary of what happened during request processing.
        
        Parameters
        ----------
        state : State
            The current state containing the user's request and API response
            
        Yields
        ------
        Dict[str, str]
            Dictionary containing the streaming summary message and preserved API response
        """
        # Store the original API response
        api_response = state.api_response
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful assistant that explains what happened during a request processing flow.
            Explain in a clear, concise way what the system did with the user's request and what the outcome was.
            Focus on the key points and use natural language that any user can understand."""),
            HumanMessagePromptTemplate.from_template(
                """User's request: {input}
                System response: {response}
                
                Please explain what happened in a clear, natural way."""
            )
        ])
        
        chain = prompt | self.model | StrOutputParser()
        full_text = ""         
        
        async for chunk in chain.astream({
            "input": state.input,
            "response": api_response
        }):
            if isinstance(chunk, AIMessageChunk):
                full_text += chunk.content
                yield {"response": full_text}


async def process_summary(state: State) -> AsyncGenerator[Dict[str, str], None]:
    """Generate a streaming summary of the process execution.
    
    Parameters
    ----------
    state : State
        The current state of the conversation
        
    Yields
    ------
    Dict[str, str]
        Dictionary containing the streaming summary message and preserved API response
    """
    handler = ProcessSummaryHandler()
    async for summary in handler.generate_summary(state):
        yield summary 