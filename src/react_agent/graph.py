"""A simple chatbot."""

"""Define the workflow for entity management and trigger creation."""
from langgraph.graph import StateGraph, END
from react_agent.state import State, InputState, OutputState
from react_agent.nodes import (
    classify_input,
    trigger_extract,
    check_geographic,
    geographic_extract,
    check_action,
    action_extract,
    query_parser,
    action_parser,
    api_agent,
    entity_name_extract,
)


def create_workflow() -> StateGraph:
    """Create the workflow graph.
    
    Returns:
        StateGraph: The workflow graph.
    """
    # Create the graph
    workflow = StateGraph(State, input=InputState, output=OutputState)
    
    # Add nodes
    workflow.add_node("classify_input", classify_input)
    workflow.add_node("trigger_extract", trigger_extract)
    workflow.add_node("check_geographic", check_geographic)
    workflow.add_node("geographic_extract", geographic_extract)
    workflow.add_node("check_action", check_action)
    workflow.add_node("action_extract", action_extract)
    workflow.add_node("entity_name_extract", entity_name_extract)
    workflow.add_node("query_parser", query_parser)
    workflow.add_node("action_parser", action_parser)
    workflow.add_node("api_agent", api_agent)

    
    # Add edges for classify_input
    workflow.add_conditional_edges(
        "classify_input",
        lambda state: state.flow_type,
        {
            "trigger": "trigger_extract",
            "action": "check_action",
            "query": "entity_name_extract",
        }
    )
    
    # Add edges for trigger flow
    workflow.add_edge("trigger_extract", "check_geographic")

    # Add conditional edges for check_geographic
    workflow.add_conditional_edges(
        "check_geographic",
        lambda state: state.is_geographic,
        {
            True: "geographic_extract",
            False: "entity_name_extract",
        }
    )

    workflow.add_edge("geographic_extract", "check_action")

    # Add conditional edges for check_action
    workflow.add_conditional_edges(
        "check_action",
        lambda state: state.action_type,
        {
            "update_entity": "action_extract",
            "create_entity": "entity_name_extract",
            "update_entity_without_query": "entity_name_extract",
        }
    )

    workflow.add_edge("action_extract", "entity_name_extract")
    workflow.add_edge("entity_name_extract", "query_parser")
    workflow.add_edge("query_parser", "action_parser")
    workflow.add_edge("action_parser", "api_agent")
    
    # Set the entry point
    workflow.set_entry_point("classify_input")
    workflow.add_edge("api_agent", END)

    
    return workflow

# Create the workflow graph
workflow = create_workflow()
# workflow.validate()

graph = workflow.compile()
# print(graph.get_graph().draw_ascii()) 
