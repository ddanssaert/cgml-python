import os
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, validator, Field, ValidationError
import yaml
import yaml_include  # Make sure this is installed
from urllib.parse import urlparse
from urllib.request import urlopen

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


def _inherit_constructor(loader: yaml.Loader, node: yaml.Node) -> Dict[str, str]:
    """YAML constructor for !inherit directive.

    Returns a sentinel mapping {"__inherit__": <path_or_url>} so we can
    detect it after parsing and perform merge with the base document.
    """
    value = loader.construct_scalar(node)
    return {"__inherit__": value}


# Add the !include constructor for PyYAML+yaml_include
yaml.add_constructor(
    "!include",
    yaml_include.Constructor(base_dir=os.path.dirname(os.path.abspath(__file__)))
)
# Add our !inherit constructor (parsed into a small mapping)
yaml.add_constructor("!inherit", _inherit_constructor)


def _read_text(path_or_url: str, base_dir: Optional[str] = None) -> str:
    """Reads text content from a local path (resolved relative to base_dir) or URL.

    Args:
        path_or_url: Local file path or http(s) URL.
        base_dir: Base directory to resolve relative paths.

    Returns:
        File contents as string.
    """
    parsed = urlparse(path_or_url)
    if parsed.scheme in {"http", "https"}:
        with urlopen(path_or_url) as resp:  # nosec - user-controlled but read-only
            return resp.read().decode("utf-8")
    # Local path
    if not os.path.isabs(path_or_url) and base_dir:
        file_path = os.path.join(base_dir, path_or_url)
    else:
        file_path = path_or_url
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _normalize_parsed_document(obj: Any) -> Dict[str, Any]:
    """Normalizes a parsed YAML document into a mapping.

    The !inherit constructor may yield a standalone mapping. Some authors may
    also structure the document as a list of top-level items. This function
    attempts to coalesce common shapes into a single mapping for further
    processing.
    """
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        merged: Dict[str, Any] = {}
        inherit_path: Optional[str] = None
        for item in obj:
            if isinstance(item, dict) and "__inherit__" in item and len(item) == 1:
                inherit_path = item["__inherit__"]
            elif isinstance(item, dict):
                merged.update(item)
        if inherit_path is not None:
            merged["__inherit__"] = inherit_path
        return merged
    # If it's just an inherit marker, wrap it
    if hasattr(obj, "get") and "__inherit__" in obj:  # type: ignore[call-arg]
        return obj  # type: ignore[return-value]
    # Fallback to empty mapping
    return {}


def _shallow_merge(parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    """Shallowly merges two dicts (one level): child values override parent."""
    merged = dict(parent or {})
    merged.update(child or {})
    return merged


def _merge_named_list(
    parent_list: Optional[List[Dict[str, Any]]],
    child_list: Optional[List[Dict[str, Any]]],
    name_key: str = "name",
) -> List[Dict[str, Any]]:
    """Merges two lists of mapping items by a stable identity key.

    - Existing parent items are preserved in order.
    - Child items with the same identity update/override parent entries (shallowly).
    - Child items with disabled: true remove the parent entry.
    - New child items are appended in the order they appear.
    """
    parent_list = list(parent_list or [])
    child_list = list(child_list or [])

    # Index parent by identity
    index: Dict[str, int] = {}
    items: List[Dict[str, Any]] = []
    for i, it in enumerate(parent_list):
        items.append(dict(it))
        if isinstance(it, dict) and name_key in it:
            index[str(it[name_key])] = i

    # Apply child changes
    appended: List[Dict[str, Any]] = []
    for it in child_list:
        if not isinstance(it, dict) or name_key not in it:
            appended.append(dict(it))  # unknown shape; append at end
            continue
        ident = str(it[name_key])
        if it.get("disabled") is True:
            # Remove if exists
            if ident in index:
                del_idx = index.pop(ident)
                # Mark removal by None placeholder; we'll filter later
                items[del_idx] = {"__REMOVED__": True}
            continue
        if ident in index:
            pos = index[ident]
            items[pos] = _shallow_merge(items[pos], it)
        else:
            appended.append(dict(it))

    # Rebuild preserving order + filtering removals
    result: List[Dict[str, Any]] = [x for x in items if not (isinstance(x, dict) and x.get("__REMOVED__"))]
    result.extend(appended)

    # Strip disabled flags for cleanliness
    for it in result:
        if isinstance(it, dict) and "disabled" in it:
            it.pop("disabled", None)
    return result


def _merge_component_types(
    parent: Optional[Dict[str, Dict[str, Any]]],
    child: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Merges component_types (e.g., deck_types, zone_types) by type/name identity."""
    parent = dict(parent or {})
    child = dict(child or {})
    merged: Dict[str, Dict[str, Any]] = {}

    all_groups = set(parent.keys()) | set(child.keys())
    for group in all_groups:
        p_group = dict(parent.get(group, {}) or {})
        c_group = dict(child.get(group, {}) or {})
        # Apply child deletions and overrides
        for name, c_val in list(c_group.items()):
            if isinstance(c_val, dict) and c_val.get("disabled") is True:
                p_group.pop(name, None)
                c_group.pop(name, None)
        # Shallow override
        merged_group: Dict[str, Any] = dict(p_group)
        for name, c_val in c_group.items():
            p_val = p_group.get(name)
            if isinstance(p_val, dict) and isinstance(c_val, dict):
                merged_group[name] = _shallow_merge(p_val, c_val)
            else:
                merged_group[name] = c_val
        merged[group] = merged_group
    return merged


def _merge_components(parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    """Merges the components block with identity-aware semantics.

    - component_types.*: by group/name
    - decks: by name (mapping keys)
    - zones: by name (list)
    - variables: by name (list)
    """
    parent = dict(parent or {})
    child = dict(child or {})
    result: Dict[str, Any] = dict(parent)

    # component_types
    if "component_types" in parent or "component_types" in child:
        result["component_types"] = _merge_component_types(
            parent.get("component_types"), child.get("component_types")
        )

    # decks (mapping)
    if "decks" in parent or "decks" in child:
        p_decks = dict(parent.get("decks", {}) or {})
        c_decks = dict(child.get("decks", {}) or {})
        # Handle deletions
        for name, c_val in list(c_decks.items()):
            if isinstance(c_val, dict) and c_val.get("disabled") is True:
                p_decks.pop(name, None)
                c_decks.pop(name, None)
        # Shallow override/add
        for name, c_val in c_decks.items():
            p_val = p_decks.get(name)
            if isinstance(p_val, dict) and isinstance(c_val, dict):
                p_decks[name] = _shallow_merge(p_val, c_val)
            else:
                p_decks[name] = c_val
        result["decks"] = p_decks

    # zones (list by name)
    if "zones" in parent or "zones" in child:
        result["zones"] = _merge_named_list(parent.get("zones"), child.get("zones"), name_key="name")

    # variables (list by name)
    if "variables" in parent or "variables" in child:
        result["variables"] = _merge_named_list(parent.get("variables"), child.get("variables"), name_key="name")

    # meta under components (shallow)
    if "meta" in child:
        result["meta"] = _shallow_merge(parent.get("meta", {}) or {}, child.get("meta", {}) or {})

    return result


def _transition_identity(tr: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Builds the identity tuple for a transition: (from, to, id?)."""
    return (
        str(tr.get("from")) if tr.get("from") is not None else None,
        str(tr.get("to")) if tr.get("to") is not None else None,
        str(tr.get("id")) if tr.get("id") is not None else None,
    )


def _merge_transitions(
    parent_list: Optional[List[Dict[str, Any]]],
    child_list: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Merges flow.transitions by (from, to, optional id) identity."""
    parent_list = list(parent_list or [])
    child_list = list(child_list or [])

    # Index parent transitions by identity
    items: List[Dict[str, Any]] = [dict(x) for x in parent_list]
    index: Dict[Tuple[Optional[str], Optional[str], Optional[str]], int] = {}
    for i, it in enumerate(items):
        index[_transition_identity(it)] = i

    appended: List[Dict[str, Any]] = []
    for tr in child_list:
        ident = _transition_identity(tr)
        if tr.get("disabled") is True:
            if ident in index:
                del_idx = index.pop(ident)
                items[del_idx] = {"__REMOVED__": True}
            else:
                # Fallback: try match by id only if provided
                if ident[2] is not None:
                    id_only = ident[2]
                    for k, pos in list(index.items()):
                        if k[2] == id_only:
                            items[pos] = {"__REMOVED__": True}
                            index.pop(k, None)
                            break
            continue
        if ident in index:
            pos = index[ident]
            items[pos] = _shallow_merge(items[pos], tr)
        else:
            appended.append(dict(tr))

    result = [x for x in items if not (isinstance(x, dict) and x.get("__REMOVED__"))]
    result.extend(appended)

    for it in result:
        it.pop("disabled", None)
    return result


def _merge_states(parent_states: Dict[str, Any], child_states: Dict[str, Any]) -> Dict[str, Any]:
    """Merges flow.states map shallowly per state name. Supports deletions via disabled: true."""
    result = dict(parent_states or {})
    child_states = dict(child_states or {})
    for name, c_val in child_states.items():
        if isinstance(c_val, dict) and c_val.get("disabled") is True:
            result.pop(name, None)
            continue
        p_val = result.get(name, {})
        if isinstance(p_val, dict) and isinstance(c_val, dict):
            result[name] = _shallow_merge(p_val, c_val)
        else:
            result[name] = c_val
    return result


def _merge_flow(parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    """Merges the flow block with identity-aware semantics for transitions.

    - states: shallow by key; deletions via disabled
    - transitions: identity by (from,to,id?)
    - other scalar keys override: initial_state, player_order
    - win_condition/meta: shallow override
    """
    parent = dict(parent or {})
    child = dict(child or {})
    result: Dict[str, Any] = dict(parent)

    # states
    if "states" in parent or "states" in child:
        result["states"] = _merge_states(parent.get("states", {}), child.get("states", {}))

    # transitions
    if "transitions" in parent or "transitions" in child:
        result["transitions"] = _merge_transitions(parent.get("transitions"), child.get("transitions"))

    # Scalars: child overrides if provided
    for k in ["initial_state", "player_order"]:
        if k in child:
            result[k] = child[k]

    # win_condition and meta shallow overrides
    if "win_condition" in child:
        result["win_condition"] = _shallow_merge(parent.get("win_condition", {}) or {}, child.get("win_condition", {}) or {})
    if "meta" in child:
        result["meta"] = _shallow_merge(parent.get("meta", {}) or {}, child.get("meta", {}) or {})

    return result


def _merge_rules(parent_list: List[Dict[str, Any]], child_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merges rules by id. Supports deletions via disabled: true.

    Preserves parent order; updates appear in place; new child rules are appended
    in their original order.
    """
    parent_list = list(parent_list or [])
    child_list = list(child_list or [])

    items: List[Dict[str, Any]] = [dict(x) for x in parent_list]
    index: Dict[str, int] = {}
    for i, it in enumerate(items):
        rid = str(it.get("id")) if it.get("id") is not None else None
        if rid:
            index[rid] = i

    appended: List[Dict[str, Any]] = []
    for rule in child_list:
        rid = str(rule.get("id")) if rule.get("id") is not None else None
        if not rid:
            appended.append(dict(rule))
            continue
        if rule.get("disabled") is True:
            if rid in index:
                del_idx = index.pop(rid)
                items[del_idx] = {"__REMOVED__": True}
            continue
        if rid in index:
            pos = index[rid]
            items[pos] = _shallow_merge(items[pos], rule)
        else:
            appended.append(dict(rule))
        # Advisory validation for extends
        ext = rule.get("extends")
        if ext and ext not in index and not any(p.get("id") == ext for p in parent_list):
            # Non-fatal note (print once). Using print to keep dependencies minimal.
            print(f"Warning: rule '{rid}' declares extends='{ext}' which was not found in parent rules.")

    result = [x for x in items if not (isinstance(x, dict) and x.get("__REMOVED__"))]
    result.extend(appended)

    for it in result:
        it.pop("disabled", None)
    return result


def _merge_top_level(parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    """Merges top-level CGML sections according to v1.3 rules.

    Sections: meta (shallow), components (custom), setup (append), flow (custom),
    rules (by id). All other keys: child overrides parent.
    """
    parent = dict(parent or {})
    child = dict(child or {})

    merged: Dict[str, Any] = dict(parent)

    # cgml_version: child overrides if present; otherwise keep parent
    if "cgml_version" in child:
        merged["cgml_version"] = child["cgml_version"]
    elif "cgml_version" in parent:
        merged["cgml_version"] = parent["cgml_version"]

    # meta
    if "meta" in parent or "meta" in child:
        merged["meta"] = _shallow_merge(parent.get("meta", {}) or {}, child.get("meta", {}) or {})

    # imports: concatenate (best-effort)
    if "imports" in parent or "imports" in child:
        merged["imports"] = list(parent.get("imports", []) or []) + list(child.get("imports", []) or [])

    # components
    if "components" in parent or "components" in child:
        merged["components"] = _merge_components(parent.get("components", {}), child.get("components", {}))

    # setup: append child's actions after parent's
    if "setup" in parent or "setup" in child:
        merged["setup"] = list(parent.get("setup", []) or []) + list(child.get("setup", []) or [])

    # flow
    if "flow" in parent or "flow" in child:
        merged["flow"] = _merge_flow(parent.get("flow", {}), child.get("flow", {}))

    # rules
    if "rules" in parent or "rules" in child:
        merged["rules"] = _merge_rules(parent.get("rules", []) or [], child.get("rules", []) or [])

    # Copy any other child keys not handled explicitly (override)
    for k, v in child.items():
        if k not in {"cgml_version", "meta", "imports", "components", "setup", "flow", "rules", "__inherit__", "inherit"}:
            merged[k] = v

    return merged


def _load_and_merge(file_path: str) -> Dict[str, Any]:
    """Loads a CGML YAML file, resolves !inherit recursively, and merges sections.

    Args:
        file_path: Path to the child CGML file.

    Returns:
        A single merged document as a Python dict.
    """
    base_dir = os.path.dirname(os.path.abspath(file_path))

    # Load raw YAML (supports !include and !inherit sentinel)
    with open(file_path, "r", encoding="utf-8") as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)

    child_doc = _normalize_parsed_document(raw)

    # Detect inherit path
    inherit_path = child_doc.pop("__inherit__", None) or child_doc.pop("inherit", None)
    if inherit_path:
        # Resolve parent (relative to current file)
        if urlparse(inherit_path).scheme in {"http", "https"}:
            parent_text = _read_text(inherit_path)
            parent_raw = yaml.load(parent_text, Loader=yaml.FullLoader)
            parent_doc = _normalize_parsed_document(parent_raw)
        else:
            parent_full = inherit_path if os.path.isabs(inherit_path) else os.path.join(base_dir, inherit_path)
            parent_doc = _load_and_merge(parent_full)
        # Merge parent -> child
        merged = _merge_top_level(parent_doc, child_doc)
        return merged

    # No inheritance; return as-is
    return child_doc


def load_cgml_file(file_path: str) -> Optional[CgmlDefinition]:
    """Loads and validates a CGML file (YAML), resolving !include and !inherit.

    Enforces cgml_version == "1.3" before model validation.
    """
    try:
        merged_doc = _load_and_merge(file_path)
    except Exception as e:
        print(f"Failed to load/merge CGML file: {e}")
        return None

    # Enforce version strictly using merged document
    ver = str(merged_doc.get("cgml_version", ""))
    if ver != "1.3":
        print(f"Validation failed: Unsupported cgml_version '{ver}'. Expected '1.3'.")
        return None

    try:
        definition = CgmlDefinition(**merged_doc)
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
