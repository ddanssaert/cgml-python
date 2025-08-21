from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import random

@dataclass
class Card:
    id: str
    name: str
    properties: Dict[str, Any]
    owner: Optional[int] = None  # player id if applicable
    face: str = "up"  # "up" or "down", default to up; set during zone assignment

@dataclass
class Zone:
    name: str
    type: str
    of_deck: Optional[str] = None
    owner: Optional[int] = None      # player id or None for shared
    ordering: Optional[str] = None
    visibility: Optional[Dict[str, str]] = None
    default_face: str = "up"
    allows_reorder: Optional[bool] = None
    cards: List[Card] = field(default_factory=list)

    @property
    def card_count(self) -> int:
        """Returns the number of cards currently in this zone."""
        return len(self.cards)

    @property
    def top_card(self) -> Optional['Card']:
        """Returns the top card in this zone, or None if zone is empty."""
        if not self.cards:
            return None
        return self.cards[-1]

    def visible_cards(self, viewer_id: Optional[int]) -> List[Card]:
        """
        Return cards visible to viewer_id according to visibility semantics on this zone.
        viewer_id: the integer id of the viewing player (or None for non-player/global)
        """
        visibility = self.visibility or {}
        if isinstance(viewer_id, int) and self.owner == viewer_id:
            vis = visibility.get("owner", visibility.get("all", "all"))
        elif viewer_id is not None:
            vis = visibility.get("others", visibility.get("all", "all"))
        else:
            vis = visibility.get("all", "all")
        if vis == "all":
            return self.cards[:]
        elif vis == "count_only":
            return []
        elif vis == "hidden":
            return []
        elif vis == "top_card_only":
            return [self.top_card] if self.top_card else []
        else:
            return self.cards[:]

@dataclass
class Player:
    id: int
    name: str
    variables: Dict[str, Any] = field(default_factory=dict)
    zones: Dict[str, Zone] = field(default_factory=dict)

@dataclass
class GameState:
    players: List[Player] = field(default_factory=list)
    shared_zones: Dict[str, Zone] = field(default_factory=dict)
    shared_variables: Dict[str, Any] = field(default_factory=dict)
    decks: Dict[str, List[Card]] = field(default_factory=dict)
    cgml_definition: Any = None     # Optionally reference to loaded CgmlDefinition
    current_state: Optional[str] = None

def create_deck(deck_def: dict, deck_type_def: Any) -> List[Card]:
    cards: List[Card] = []
    comp = deck_type_def.composition if hasattr(deck_type_def, "composition") else []
    idx = 0
    for entry in comp:
        if entry.get("type") == "template" and entry.get("template") == "standard_suits":
            ranks = entry.get("values")
            suits = ["♠", "♥", "♦", "♣"]
            for suit in suits:
                for rank in ranks:
                    idx += 1
                    card_id = f"{deck_def['type']}-{suit}-{rank}-{idx}"
                    card_name = f"{rank}{suit}"
                    cards.append(Card(
                        id=card_id,
                        name=card_name,
                        properties={"rank": rank, "suit": suit}
                    ))
    return cards

def build_game_state_from_cgml(cgml: Any, player_count: int = None) -> GameState:
    deck_types = cgml.components.component_types.get('deck_types', {}) if cgml.components.component_types else {}
    zone_types = cgml.components.component_types.get('zone_types', {}) if cgml.components.component_types else {}
    decks: Dict[str, List[Card]] = {}
    for deck_name, deck_def in (cgml.components.decks or {}).items():
        deck_type_def = deck_types.get(deck_def.type, {})
        decks[deck_name] = create_deck(deck_def.dict(), deck_type_def)

    player_count = player_count or cgml.meta.players.max
    players: List[Player] = []
    var_defs = cgml.components.variables if cgml.components.variables else []
    per_player_vars = {v.name: v.initial_value for v in var_defs if v.per_player}
    shared_vars = {v.name: v.initial_value for v in var_defs if not v.per_player}

    zone_defs = cgml.components.zones or []
    shared_zones: Dict[str, Zone] = {}
    per_player_zone_defs = [z for z in zone_defs if getattr(z, "per_player", False)]
    shared_zone_defs = [z for z in zone_defs if not getattr(z, "per_player", False)]

    def _apply_zone_type(zone: Zone) -> None:
        # Copy ordering/visibility/default_face/allows_reorder from zone_types if available
        zt = zone_types.get(zone.type, None)
        if zt is not None:
            if getattr(zt, 'ordering', None) is not None:
                zone.ordering = zt.ordering
            if getattr(zt, 'visibility', None) is not None:
                zone.visibility = zt.visibility
            if getattr(zt, 'default_face', None) is not None:
                zone.default_face = zt.default_face
            if getattr(zt, 'allows_reorder', None) is not None:
                zone.allows_reorder = zt.allows_reorder

    for pidx in range(player_count):
        pname = f"Player {pidx + 1}"
        player = Player(id=pidx, name=pname)
        player.variables = {k: v for k, v in per_player_vars.items()}
        for zone_def in per_player_zone_defs:
            zone = Zone(
                name=zone_def.name,
                type=zone_def.type,
                of_deck=getattr(zone_def, 'of_deck', None),
                owner=pidx
            )
            _apply_zone_type(zone)
            player.zones[zone.name] = zone
        players.append(player)

    for zone_def in shared_zone_defs:
        zone = Zone(
            name=zone_def.name,
            type=zone_def.type,
            of_deck=getattr(zone_def, 'of_deck', None),
            owner=None
        )
        _apply_zone_type(zone)
        shared_zones[zone.name] = zone

    state = GameState(
        players=players,
        shared_zones=shared_zones,
        shared_variables=shared_vars,
        decks=decks,
        cgml_definition=cgml,
    )

    # Assign the cards (by reference!) from deck into the appropriate zone at setup.
    # For each assignment, set card.face to the zone's default_face.
    for deck_name, cards in decks.items():
        assigned = False
        for zone in state.shared_zones.values():
            if getattr(zone, 'of_deck', None) == deck_name:
                for c in cards:
                    c.face = zone.default_face
                zone.cards.extend(cards)
                assigned = True
        for player in state.players:
            for zone in player.zones.values():
                if getattr(zone, 'of_deck', None) == deck_name:
                    for c in cards:
                        c.face = zone.default_face
                    zone.cards.extend(cards)
                    assigned = True
        if not assigned:
            print(f"Warning: Deck '{deck_name}' was generated but not assigned to any zone!")

    # Shuffle zones that declare ordering: shuffled
    for zone in list(state.shared_zones.values()) + [z for p in state.players for z in p.zones.values()]:
        if getattr(zone, 'ordering', None) == 'shuffled':
            random.shuffle(zone.cards)

    return state

def find_zone(state: GameState, zone_path: str, player: Optional[Player] = None) -> Zone:
    """
    Resolves a selector-like path to a Zone within the GameState.
    - Supports: '$.players[0].zones.discard', '$.zones.deck'
    - Minimal: '$.players[$player].zones.<name>' with context injection handled elsewhere.
    """
    if isinstance(zone_path, Zone):
        return zone_path
    if not isinstance(zone_path, str) or not zone_path.startswith('$.'):
        raise ValueError("Only $-rooted selector paths are supported for zone lookups.")
    current: Any = {
        'players': state.players,
        'zones': state.shared_zones,
    }
    parts: List[str] = []
    buf = ''
    i = 2
    while i < len(zone_path):
        ch = zone_path[i]
        if ch == '.' and '[' not in buf and ']' not in buf:
            if buf:
                parts.append(buf)
                buf = ''
            i += 1
            continue
        buf += ch
        i += 1
    if buf:
        parts.append(buf)
    for part in parts:
        key = part
        idx: Optional[int] = None
        if '[' in part and part.endswith(']'):
            key = part[: part.index('[')]
            inside = part[part.index('[') + 1 : -1]
            if inside != '*':
                idx = int(inside)
        if key:
            if isinstance(current, dict):
                current = current[key]
            else:
                current = getattr(current, key)
        if idx is not None:
            current = current[idx]
    if isinstance(current, Zone):
        return current
    raise ValueError(f"Path '{zone_path}' does not resolve to a Zone")

def shuffle_zone(zone: Zone) -> None:
    random.shuffle(zone.cards)

def move_cards(from_zone: Zone, to_zone: Zone, count: int = 1) -> None:
    for _ in range(min(count, len(from_zone.cards))):
        # Remove from source zone according to its ordering
        if from_zone.ordering == "fifo":
            card = from_zone.cards.pop(0)
        else:  # lifo/shuffled/unordered/default
            card = from_zone.cards.pop()
        # Insert into target zone according to its ordering
        if to_zone.ordering == "fifo":
            to_zone.cards.append(card)
        elif to_zone.ordering == "lifo" or to_zone.ordering == "shuffled":
            to_zone.cards.append(card)
        elif to_zone.ordering == "unordered":
            to_zone.cards.append(card)  # unordered: just append
        else:
            to_zone.cards.append(card)

def move_all_cards(from_zone: Zone, to_zone: Zone) -> None:
    while from_zone.cards:
        move_cards(from_zone, to_zone, 1)

def deal_cards(from_zone: Zone, players: List[Player], to_zone_name: str, count: int) -> None:
    for _ in range(count):
        for player in players:
            if from_zone.cards:
                player.zones[to_zone_name].cards.append(from_zone.cards.pop())

def deal_all_cards(from_deck: Zone, players: List[Player], to_zone_name: str) -> None:
    idx = 0
    pl_count = len(players)
    while from_deck.cards:
        player = players[idx % pl_count]
        player.zones[to_zone_name].cards.append(from_deck.cards.pop())
        idx += 1

def find_card_zone(state: GameState, card: Card) -> Optional[Zone]:
    """Locate the zone containing the given card by identity or id."""
    for zone in state.shared_zones.values():
        for c in zone.cards:
            if c is card or c.id == card.id:
                return zone
    for p in state.players:
        for zone in p.zones.values():
            for c in zone.cards:
                if c is card or c.id == card.id:
                    return zone
    return None

def card_owner(card: Card, state: GameState) -> Optional[int]:
    """
    Returns the owner/player id for the card, if present, based on containing zone
    and owner/owner_scope semantics (v1.3).
    """
    zone = find_card_zone(state, card)
    if not zone:
        return None
    return zone.owner

def perform_setup_action(action: dict, state: GameState) -> None:
    """Perform a single setup action (supports v1.3 path operands for basic actions)."""
    typ = action['action']
    if typ == "SHUFFLE":
        target = action.get("target")
        if isinstance(target, dict) and 'path' in target:
            zone = find_zone(state, target['path'])
            shuffle_zone(zone)
        elif isinstance(target, str):
            zone = find_zone(state, target)
            shuffle_zone(zone)
        else:
            raise ValueError("SHUFFLE requires target.path")
    elif typ == "DEAL":
        frm = action.get('from')
        to = action.get('to')
        count = int(action.get('count', 1))
        from_zone = find_zone(state, frm['path'] if isinstance(frm, dict) else frm)
        to_path = to['path'] if isinstance(to, dict) else to
        if not to_path.startswith('$.players['):
            raise ValueError("DEAL setup expects a specific player's zone path under $.players[<idx>].zones.<name>")
        try:
            idx_start = to_path.index('[') + 1
            idx_end = to_path.index(']')
            pidx = int(to_path[idx_start:idx_end])
            zone_name = to_path.split('.zones.')[1]
        except Exception as e:
            raise ValueError(f"Invalid DEAL target path: {to_path}") from e
        for _ in range(count):
            if from_zone.cards:
                state.players[pidx].zones[zone_name].cards.append(from_zone.cards.pop())
    elif typ == "DEAL_ROUND_ROBIN":
        frm = action.get('from')
        to = action.get('to')
        order = (action.get('order') or 'clockwise').lower()
        count = int(action.get('count', 1))
        from_zone = find_zone(state, frm['path'] if isinstance(frm, dict) else frm)
        to_path = to['path'] if isinstance(to, dict) else to
        if '[*]' not in to_path or '.zones.' not in to_path:
            raise ValueError("DEAL_ROUND_ROBIN setup expects path like $.players[*].zones.<zone>")
        to_zone_name = to_path.split('.zones.')[-1]
        targets = list(state.players)
        if order == 'counterclockwise':
            targets = list(reversed(targets))
        deal_cards(from_zone, targets, to_zone_name, count)
    elif typ == "MOVE":
        frm = action.get('from')
        to = action.get('to')
        count = action.get('count', 1)
        from_zone = find_zone(state, frm['path'] if isinstance(frm, dict) else frm)
        to_zone = find_zone(state, to['path'] if isinstance(to, dict) else to)
        move_cards(from_zone, to_zone, count)
    elif typ == "MOVE_ALL":
        frm = action.get('from')
        to = action.get('to')
        from_zone = find_zone(state, frm['path'] if isinstance(frm, dict) else frm)
        to_zone = find_zone(state, to['path'] if isinstance(to, dict) else to)
        move_all_cards(from_zone, to_zone)
    elif typ == "DEAL_ALL":
        frm = action.get('from')
        to = action.get('to')
        from_path = frm['path'] if isinstance(frm, dict) else frm
        to_path = to['path'] if isinstance(to, dict) else to
        deck_zone = find_zone(state, from_path)
        to_zone_name = to_path.split('.zones.')[-1]
        deal_all_cards(deck_zone, state.players, to_zone_name)
    else:
        raise NotImplementedError(f"Unknown setup action: {typ}")

def run_setup_phase(state: GameState) -> None:
    """Runs all setup actions from the CGML file."""
    for action in state.cgml_definition.setup:
        if hasattr(action, "dict"):
            d = action.dict(by_alias=True, exclude_none=True)
        else:
            d = dict(action)
        perform_setup_action(d, state)
