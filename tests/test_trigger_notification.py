"""Tests for trigger notification handler."""

import pytest
from react_agent.nodes.trigger_notification import TriggerNotificationHandler
from react_agent.configuration import Configuration


@pytest.fixture
def notification_handler():
    """Create a notification handler for testing."""
    config = Configuration(model="gpt-4")  # or any other suitable test model
    return TriggerNotificationHandler(config)


@pytest.mark.asyncio
async def test_process_geo_trigger_event(notification_handler):
    """Test processing a geographic trigger event."""
    # Test data for geographic trigger
    test_data = {
        "trigger_data": {
            "id": "trigger123",
            "type": "geoRule",
            "sourceQuery": {
                "type": "vehicle",
                "area": "North Tel Aviv"
            },
            "targetQuery": {
                "type": "parking",
                "status": "available"
            },
            "actions": [{
                "type": "createEntity",
                "payload": {"type": "task", "status": "pending"}
            }]
        },
        "event_data": {
            "entity_type": "vehicle",
            "entity_id": "v123",
            "location": {
                "area": "North Tel Aviv",
                "coordinates": {"lat": 32.0853, "lon": 34.7818}
            },
            "timestamp": "2024-03-20T10:30:00Z"
        }
    }
    
    result = await notification_handler.process_trigger_event(test_data)
    
    assert result is not None
    assert "response" in result
    assert "messages" in result
    assert len(result["messages"]) == 1
    
    # Check that message contains relevant information
    message_content = result["messages"][0].content
    assert "North Tel Aviv" in message_content
    assert "vehicle" in message_content
    assert "task" in message_content


@pytest.mark.asyncio
async def test_process_layer_trigger_event(notification_handler):
    """Test processing a layer trigger event."""
    # Test data for layer trigger
    test_data = {
        "trigger_data": {
            "id": "trigger456",
            "type": "layer",
            "query": {
                "type": "sensor",
                "status": "active",
                "threshold": 100
            },
            "actions": [{
                "type": "updateEntity",
                "payload": {"status": "alert"},
                "query": {"type": "device", "id": "d789"}
            }]
        },
        "event_data": {
            "entity_type": "sensor",
            "entity_id": "s456",
            "reading": 150,
            "timestamp": "2024-03-20T10:35:00Z"
        }
    }
    
    result = await notification_handler.process_trigger_event(test_data)
    
    assert result is not None
    assert "response" in result
    assert "messages" in result
    assert len(result["messages"]) == 1
    
    # Check that message contains relevant information
    message_content = result["messages"][0].content
    assert "sensor" in message_content
    assert "150" in message_content
    assert "alert" in message_content


@pytest.mark.asyncio
async def test_invalid_trigger_data(notification_handler):
    """Test handling invalid trigger data."""
    # Invalid test data
    invalid_test_data = {
        "trigger_data": {},
        "event_data": {}
    }
    
    with pytest.raises(Exception):
        await notification_handler.process_trigger_event(invalid_test_data) 