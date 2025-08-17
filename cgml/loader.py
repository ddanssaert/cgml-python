import os
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, validator, Field, ValidationError
import yaml
import yaml_include  # Make sure this is installed

# ---- Data Models ---- #

class PlayerSpec(BaseModel):
    min: int
    max: int

class RNGConfig(BaseModel):
    """Randomness configuration for determinism and seeding."""
    deterministic: bool = False
    seed: Optional[int] = None


class Meta(BaseModel):
    name: str
    author: str
    description: str
    players: PlayerSpec
    rng: Optional[RNGConfig] = None
    meta: Optional[Dict[str, Any]] = None  # for extensibility/docs

# -- Components, Decks, Zones, Variables -- #

class ComponentTypeDef(BaseModel):
    # Extensible for decks/zones/others
    composition: Optional[List[Any]] = None
    ordering: Optional[str] = None
    visibility: Optional[Dict[str, Any]] = None
    rank_hierarchy: Optional[List[Union[str, int]]] = None

class DeckInstance(BaseModel):
    type: str
    meta: Optional[Dict[str, Any]] = None  # extensibility

class ZoneInstance(BaseModel):
    name: str
    type: str
    of_deck: Optional[str] = None
    per_player: Optional[bool] = False
    # Optionally add owners/scopes in future extension
    meta: Optional[Dict[str, Any]] = None

class VariableInstance(BaseModel):
    name: str
    per_player: Optional[bool] = False
    initial_value: Optional[Any] = None
    computed: Optional[Any] = None  # complex: use expr-model (future)

class Components(BaseModel):
    component_types: Optional[Dict[str, Dict[str, ComponentTypeDef]]] = None  # deck_types, zone_types, etc
    decks: Optional[Dict[str, DeckInstance]] = None
    zones: Optional[List[ZoneInstance]] = None
    variables: Optional[List[VariableInstance]] = None
    meta: Optional[Dict[str, Any]] = None

# --- Action/Setup Model --- #

class Action(BaseModel):
    action: str
    # Flexible: allow arbitrary keys
    params: Optional[Dict[str, Any]] = None
    # For atomic fields (common actions)
    from_: Optional[Any] = Field(None, alias="from")
    from_deck: Optional[str] = None
    to: Optional[Any] = None
    target: Optional[Any] = None
    player: Optional[Any] = None
    prompt: Optional[str] = None
    store_as: Optional[str] = None
    value: Optional[Any] = None
    condition: Optional[Any] = None

    class Config:
        extra = "allow"

# --- Condition & Expression System (Recursive) --- #

# Expression operand could be:
#   { path: ... } | { value: ... } | any nested expression
class Operand(BaseModel):
    path: Optional[str] = None
    value: Optional[Any] = None
    ref: Optional[str] = None  # For referencing stored values
    # Nested operator: allows recursion (for lists/maps/etc)
    isEqual: Optional[List["Operand"]] = None
    isGreaterThan: Optional[List["Operand"]] = None
    isLessThan: Optional[List["Operand"]] = None
    and_: Optional[List["Operand"]] = Field(None, alias="and")
    or_: Optional[List["Operand"]] = Field(None, alias="or")
    not_: Optional["Operand"] = Field(None, alias="not")
    any_: Optional[List["Operand"]] = Field(None, alias="any")
    all_: Optional[List["Operand"]] = Field(None, alias="all")
    max_: Optional[List["Operand"]] = Field(None, alias="max")
    min_: Optional[List["Operand"]] = Field(None, alias="min")
    sum_: Optional[List["Operand"]] = Field(None, alias="sum")
    count: Optional[List["Operand"]] = None
    # For filters/maps/group_by/etc
    distinct: Optional[List["Operand"]] = None
    filter: Optional[List["Operand"]] = None
    map: Optional[List["Operand"]] = None
    group_by: Optional[Any] = None  # Groupings for forEach/books, etc
    having: Optional[Any] = None

    # v1.3 operators (selective support for this project)
    rank_value: Optional[List["Operand"]] = None
    top: Optional[List["Operand"]] = None
    all_items: Optional[List["Operand"]] = Field(None, alias="all")
    add: Optional[List["Operand"]] = None
    list_: Optional[List["Operand"]] = Field(None, alias="list")

    def is_leaf(self) -> bool:
        """Returns True if this is a terminal (path or value)."""
        return (self.path is not None or self.value is not None or self.ref is not None)

    class Config:
        extra = 'allow'
        arbitrary_types_allowed = True

Operand.update_forward_refs()
class Condition(BaseModel):
    # Only one key should be set per node, but we allow all for composition.
    isEqual: Optional[List[Operand]] = None
    isGreaterThan: Optional[List[Operand]] = None
    isLessThan: Optional[List[Operand]] = None
    and_: Optional[List["Condition"]] = Field(None, alias="and")
    or_: Optional[List["Condition"]] = Field(None, alias="or")
    not_: Optional["Condition"] = Field(None, alias="not")
    any_: Optional[List["Condition"]] = Field(None, alias="any")
    all_: Optional[List["Condition"]] = Field(None, alias="all")
    max_: Optional[List[Operand]] = Field(None, alias="max")
    min_: Optional[List[Operand]] = Field(None, alias="min")
    sum_: Optional[List[Operand]] = Field(None, alias="sum")
    count: Optional[List[Operand]] = None
    filter: Optional[List["Condition"]] = None

    # For referencing sub-branches (group_by, forEach, etc.)
    distinct: Optional[List[Operand]] = None
    group_by: Optional[Any] = None
    having: Optional[Any] = None

    class Config:
        extra = "allow"

Condition.update_forward_refs()

# ---- Flow (FSM) ---- #
class Transition(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    condition: Optional[Condition] = None

class StateDef(BaseModel):
    phases: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None

class Flow(BaseModel):
    states: Dict[str, StateDef]  # <-- changed from List[str] to Dict[str, StateDef]
    initial_state: str
    player_order: str  # 'clockwise', 'counterclockwise', 'simultaneous'
    transitions: Optional[List[Transition]] = Field(default_factory=list)
    win_condition: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None

# --- Rule System --- #

class EffectAction(BaseModel):
    action: str
    # Allow arbitrary key/values for extensibility
    params: Optional[Dict[str, Any]] = None
    from_: Optional[Any] = Field(None, alias="from")
    to: Optional[Any] = None
    player: Optional[Any] = None
    target: Optional[Any] = None
    count: Optional[Union[int, Dict[str, Any]]] = None
    filter: Optional[Any] = None
    value: Optional[Any] = None
    prompt: Optional[str] = None
    store_as: Optional[str] = None
    condition: Optional[Any] = None
    state: Optional[str] = None

    class Config:
        extra = "allow"

class Rule(BaseModel):
    id: str
    trigger: str
    condition: Optional[Condition] = None
    effect: Optional[List[EffectAction]] = None
    description: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

# --- Top-level CGML Object --- #

class CgmlDefinition(BaseModel):
    cgml_version: Union[float, str]  # v1.2+
    meta: Meta
    imports: Optional[List[Any]] = None
    components: Components
    setup: List[Action]
    flow: Flow
    rules: List[Rule]
# ---- Loader ---- #

# Add the !include constructor for PyYAML+yaml_include
yaml.add_constructor(
    "!include",
    yaml_include.Constructor(base_dir=os.path.dirname(os.path.abspath(__file__)))
)

def load_cgml_file(file_path: str) -> Optional[CgmlDefinition]:
    """Loads and validates a CGML file (YAML), resolving all !include directives.

    Enforces cgml_version == "1.3" before model validation.
    """
    with open(file_path, "r") as f:
        dct = yaml.load(f, Loader=yaml.FullLoader)

    # Enforce version strictly
    ver = str(dct.get("cgml_version", ""))
    if ver != "1.3":
        print(f"Validation failed: Unsupported cgml_version '{ver}'. Expected '1.3'.")
        return None

    try:
        definition = CgmlDefinition(**dct)
        return definition
    except ValidationError as e:
        print("Validation failed:")
        print(e)
        return None

# Optionally add further utilities/methods as needed.

if __name__ == "__main__":
    import json

    # Generate the JSON schema for the top-level CgmlDefinition model
    schema = CgmlDefinition.model_json_schema()  # For Pydantic v2+
    # If using Pydantic v1.x, use: schema = CgmlDefinition.schema()

    with open("../cgml-specification/schema/cgml.schema.json", "w") as f:
        json.dump(schema, f, indent=2)
