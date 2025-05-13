from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Literal, Union
import uvicorn
from datetime import datetime
import uuid

app = FastAPI(title="Mock Entity Management Server")

# --- Models ---
class Position(BaseModel):
    lat: float
    lng: float

class EntityBase(BaseModel):
    type: str
    properties: Dict[str, Any]
    position: Optional[Position] = None

class Action(BaseModel):
    type: Literal["updateEntity", "createEntity"]
    payload: Dict[str, Any]
    query: Dict[str, Any]

class GeoRuleTrigger(BaseModel):
    type: Literal["geoRule"]
    sourceQuery: Dict[str, Any]
    targetQuery: Dict[str, Any]
    actions: List[Action]

class QueryRuleTrigger(BaseModel):
    type: Literal["queryRule"]
    sourceQuery: Dict[str, Any]
    actions: List[Action]

Trigger = Union[GeoRuleTrigger, QueryRuleTrigger]

# --- In-memory storage ---
entities: Dict[str, List[Dict[str, Any]]] = {
    "target": [],
    "isrtask": [],
    "arrow": [],
    "friendlyforce": [],
    "phaseline": [],
    "casevactask": [],
    "firemission": []
}

triggers: List[Dict[str, Any]] = []

# --- Helper functions ---
def generate_id() -> str:
    return str(uuid.uuid4())

def get_current_time() -> datetime:
    return datetime.utcnow()

# --- Entity endpoints ---
@app.get("/entities/{entity_type}")
async def get_entities(entity_type: str, filters: Optional[Dict[str, Any]] = None):
    """Get entities of a specific type with optional filters."""
    entity_type = entity_type.lower()
    if entity_type not in entities:
        raise HTTPException(status_code=404, detail=f"Entity type {entity_type} not found")
    
    if not filters:
        return {"entities": entities[entity_type]}
    
    # Simple filter implementation
    filtered_entities = []
    for entity in entities[entity_type]:
        matches = True
        for key, value in filters.items():
            if key not in entity or entity[key] != value:
                matches = False
                break
        if matches:
            filtered_entities.append(entity)
    
    return {"entities": filtered_entities}

@app.post("/entities/add")
async def create_entity(entity: EntityBase):
    """Create a new entity."""
    entity_type = entity.type.lower()
    if entity_type not in entities:
        raise HTTPException(status_code=404, detail=f"Entity type {entity_type} not found")
    
    # Add metadata
    entity_data = {
        "id": generate_id(),
        "type": entity.type,
        "properties": entity.properties,
        "created_at": get_current_time(),
        "updated_at": get_current_time()
    }
    
    # Add position if provided
    if entity.position:
        entity_data["position"] = entity.position.dict()
    
    entities[entity_type].append(entity_data)
    return {"entity": entity_data}

@app.put("/entities/update/{entity_id}")
async def update_entity(entity_id: str, update_data: Dict[str, Any]):
    """Update an existing entity."""
    # Find entity in any type
    for entity_type, entity_list in entities.items():
        for i, entity in enumerate(entity_list):
            if entity["id"] == entity_id:
                # Update properties
                if "properties" in update_data:
                    entity["properties"].update(update_data["properties"])
                
                # Update position if provided
                if "position" in update_data:
                    entity["position"] = update_data["position"]
                
                entity["updated_at"] = get_current_time()
                return {"entity": entity}
    
    raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

@app.get("/entities/query")
async def query_entities(request: Request):
    """Query entities with type parameter."""
    # Convert QueryParams to dict and handle type conversion
    params = {}
    for key, value in request.query_params.items():
        # Try to convert to float if possible
        try:
            params[key] = float(value)
        except ValueError:
            params[key] = value
    
    entity_type = params.get("type", "target").lower()
    
    if entity_type not in entities:
        raise HTTPException(status_code=404, detail=f"Entity type {entity_type} not found")
    
    # For demo purposes, return all entities of the type
    return {"entities": entities[entity_type]}

# --- Trigger endpoints ---
@app.get("/triggers")
async def get_triggers():
    """Get all triggers."""
    return {"triggers": triggers}

@app.post("/triggers/new")
async def create_trigger(trigger: Trigger):
    """Create a new trigger."""
    trigger_data = trigger.dict()
    trigger_data["id"] = generate_id()
    trigger_data["created_at"] = get_current_time()
    trigger_data["updated_at"] = get_current_time()
    
    triggers.append(trigger_data)
    return {"trigger": trigger_data}

@app.put("/triggers/{trigger_id}")
async def update_trigger(trigger_id: str, trigger: Trigger):
    """Update an existing trigger."""
    for i, existing_trigger in enumerate(triggers):
        if existing_trigger["id"] == trigger_id:
            trigger_data = trigger.dict()
            trigger_data["id"] = trigger_id
            trigger_data["created_at"] = existing_trigger["created_at"]
            trigger_data["updated_at"] = get_current_time()
            triggers[i] = trigger_data
            return {"trigger": trigger_data}
    
    raise HTTPException(status_code=404, detail=f"Trigger {trigger_id} not found")

@app.delete("/triggers/{trigger_id}")
async def delete_trigger(trigger_id: str):
    """Delete a trigger."""
    for i, trigger in enumerate(triggers):
        if trigger["id"] == trigger_id:
            triggers.pop(i)
            return {"status": "success", "message": f"Trigger {trigger_id} deleted"}
    
    raise HTTPException(status_code=404, detail=f"Trigger {trigger_id} not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5005, reload=True) 