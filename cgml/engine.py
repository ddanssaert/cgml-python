from typing import Any, Dict, Callable, List, Union, Optional

from loader import Condition, Operand, EffectAction


def resolve_path(obj: Any, path: str, context: Optional[Dict[str, Any]] = None) -> Any:
    """
    Selector resolver supporting a subset of CGML v1.3 $-rooted syntax:
    - $.players[0].zones.hand
    - $.zones.deck
    - $.players[*]
    - $.players[$player]

    Non-$ dotted paths are not supported.
    """
    context = context or {}
    if path is None:
        return obj

    p = path.strip()
    if not p.startswith("$"):
        raise ValueError(f"Only $-rooted selector paths are supported (got: {path})")

    # Build a synthetic root for selector-style paths
    root = {
        "players": getattr(obj, "players", None),
        "zones": getattr(obj, "shared_zones", None),
    }
    current: Any = root
    # Tokenize by dots while keeping bracket segments grouped
    parts: List[str] = []
    buf = ""
    i = 1  # skip leading $
    while i < len(p):
        ch = p[i]
        if ch == '.' and ('[' not in buf or (buf.count('[') == buf.count(']'))):
            if buf:
                parts.append(buf)
                buf = ""
            i += 1
            continue
        buf += ch
        i += 1
    if buf:
        parts.append(buf)
    # Resolve each part (with optional [index])
    for part in parts:
        key = part
        idx_token: Optional[str] = None
        star = False
        if '[' in part and part.endswith(']'):
            key = part[: part.index('[')]
            inside = part[part.index('[') + 1 : -1]
            if inside == '*':
                star = True
            else:
                idx_token = inside
        if key:
            if isinstance(current, list):
                # map attribute/key over list elements
                mapped: List[Any] = []
                for elem in current:
                    if isinstance(elem, dict):
                        mapped.append(elem.get(key))
                    else:
                        mapped.append(getattr(elem, key))
                current = mapped
            elif isinstance(current, dict):
                current = current.get(key)
            else:
                current = getattr(current, key)
        if star:
            # Keep list as-is
            if isinstance(current, dict):
                current = list(current.values())
            if not isinstance(current, list):
                current = [current]
        elif idx_token is not None:
            # Index can be integer or $player reference
            if idx_token.startswith('$'):
                if idx_token in context:
                    try:
                        idx = int(context[idx_token])
                    except Exception:
                        idx = context[idx_token]
                else:
                    raise KeyError(f"Context variable '{idx_token}' not set for path {path}")
            else:
                try:
                    idx = int(idx_token)
                except ValueError:
                    idx = idx_token
            if isinstance(current, list):
                current = current[idx]  # type: ignore[index]
            elif isinstance(current, dict):
                current = current[idx]  # type: ignore[index]
            else:
                raise KeyError(f"Cannot index non-collection with [{idx_token}] in path {path}")
    return current


def get_rank_index(cgml_definition: Any, deck_type_name: str, rank_value: Any) -> int:
    """Looks up the numeric index of a rank in the deck's rank_hierarchy."""
    rank_hierarchy = cgml_definition.components.component_types['deck_types'][deck_type_name].rank_hierarchy
    try:
        return [str(x) for x in rank_hierarchy].index(str(rank_value))
    except ValueError:
        raise ValueError(f"Rank '{rank_value}' not found in rank_hierarchy: {rank_hierarchy}")


class RulesEngine:
    """Evaluates conditions and executes effects using an action registry."""

    def __init__(self, action_registry: Dict[str, Callable]):
        self.actions = action_registry

    def _maybe_compare_ranks(self, left: Any, right: Any, game_state: Any) -> (Any, Any):
        """Convert left/right to comparable ordinal if they look like ranks for the game's deck type."""
        cgml_def = getattr(game_state, "cgml_definition", None)
        if cgml_def is None:
            return left, right
        try:
            deck_types = cgml_def.components.component_types.get('deck_types', {})
            if not deck_types:
                return left, right
            deck_type_name = next(iter(deck_types))
            rank_hierarchy = [str(x) for x in deck_types[deck_type_name].rank_hierarchy]
            if str(left) in rank_hierarchy and str(right) in rank_hierarchy:
                left = rank_hierarchy.index(str(left))
                right = rank_hierarchy.index(str(right))
        except Exception:
            pass
        return left, right

    def _op_rank_value(self, arg: Any, game_state: Any, context: Dict[str, Any]) -> int:
        value = self.resolve_operand(arg, game_state, context)
        try:
            if hasattr(value, 'properties') and isinstance(value.properties, dict) and 'rank' in value.properties:
                rank = value.properties['rank']
            elif isinstance(value, dict) and 'properties' in value and 'rank' in value['properties']:
                rank = value['properties']['rank']
            else:
                rank = value
        except Exception:
            rank = value
        cgml_def = getattr(game_state, "cgml_definition", None)
        if cgml_def and cgml_def.components and cgml_def.components.component_types:
            deck_types = cgml_def.components.component_types.get('deck_types', {})
            if deck_types:
                deck_type_name = next(iter(deck_types))
                return get_rank_index(cgml_def, deck_type_name, rank)
        order = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
        return order.index(str(rank)) if str(rank) in order else int(rank)

    @staticmethod
    def _count_value(value: Any) -> int:
        """Best-effort counting for zones, lists, strings, and scalars."""
        if value is None:
            return 0
        # Zone-like with cards
        if hasattr(value, 'cards') and isinstance(getattr(value, 'cards'), list):
            return len(getattr(value, 'cards'))
        if hasattr(value, 'card_count'):
            try:
                return int(getattr(value, 'card_count'))
            except Exception:
                pass
        # Collections and strings
        try:
            return len(value)
        except TypeError:
            return 1

    def evaluate_condition(
        self,
        cond: Union[Condition, Dict, Any],
        game_state: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Recursively evaluates a Condition (pydantic model or dict node)."""
        context = context or {}

        if not isinstance(cond, (Condition, dict)):
            return bool(cond)

        if isinstance(cond, dict):
            cond = Condition.parse_obj(cond)

        if cond.isEqual is not None:
            left = self.resolve_operand(cond.isEqual[0], game_state, context)
            right = self.resolve_operand(cond.isEqual[1], game_state, context)
            left, right = self._maybe_compare_ranks(left, right, game_state)
            return left == right
        if cond.isGreaterThan is not None:
            left = self.resolve_operand(cond.isGreaterThan[0], game_state, context)
            right = self.resolve_operand(cond.isGreaterThan[1], game_state, context)
            left, right = self._maybe_compare_ranks(left, right, game_state)
            return left > right
        if cond.isLessThan is not None:
            left = self.resolve_operand(cond.isLessThan[0], game_state, context)
            right = self.resolve_operand(cond.isLessThan[1], game_state, context)
            left, right = self._maybe_compare_ranks(left, right, game_state)
            return left < right
        if getattr(cond, "and_", None) is not None:
            return all(self.evaluate_condition(sub, game_state, context) for sub in cond.and_)
        if getattr(cond, "or_", None) is not None:
            return any(self.evaluate_condition(sub, game_state, context) for sub in cond.or_)
        if getattr(cond, "not_", None) is not None:
            return not self.evaluate_condition(cond.not_, game_state, context)
        if getattr(cond, "max_", None) is not None:
            vals = [self.resolve_operand(x, game_state, context) for x in cond.max_]
            return max(vals)
        if getattr(cond, "min_", None) is not None:
            vals = [self.resolve_operand(x, game_state, context) for x in cond.min_]
            return min(vals)
        if getattr(cond, "sum_", None) is not None:
            vals = [self.resolve_operand(x, game_state, context) for x in cond.sum_]
            flat_vals: List[Any] = []
            for v in vals:
                if isinstance(v, (list, tuple)):
                    flat_vals.extend(v)
                else:
                    flat_vals.append(v)
            return sum(flat_vals)
        if getattr(cond, "count", None) is not None:
            collection = cond.count
            if isinstance(collection, list) and len(collection) == 1:
                resolved = self.resolve_operand(collection[0], game_state, context)
                return self._count_value(resolved)
            elif isinstance(collection, (list, tuple)):
                return len(collection)
            else:
                return self._count_value(collection)
        if getattr(cond, "value", None) is not None:
            return bool(cond.value)
        if getattr(cond, "path", None) is not None:
            return bool(resolve_path(game_state, cond.path, context))
        if getattr(cond, "ref", None) is not None:
            return bool(context.get(cond.ref))
        return False

    def resolve_operand(
        self,
        operand: Union[Operand, Dict, Any],
        game_state: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Resolves an operand node: can be Operand model, dict, or value."""
        context = context or {}

        if isinstance(operand, dict):
            operand = Operand.parse_obj(operand)
        if isinstance(operand, Operand):
            if operand.path is not None:
                return resolve_path(game_state, operand.path, context)
            if operand.value is not None:
                return operand.value
            if operand.ref is not None:
                return context.get(operand.ref)
            if operand.isEqual is not None:
                return self.evaluate_condition(Condition(isEqual=operand.isEqual), game_state, context)
            if operand.isGreaterThan is not None:
                return self.evaluate_condition(Condition(isGreaterThan=operand.isGreaterThan), game_state, context)
            if operand.isLessThan is not None:
                return self.evaluate_condition(Condition(isLessThan=operand.isLessThan), game_state, context)
            if getattr(operand, "and_", None) is not None:
                return self.evaluate_condition(Condition(and_=operand.and_), game_state, context)
            if getattr(operand, "or_", None) is not None:
                return self.evaluate_condition(Condition(or_=operand.or_), game_state, context)
            if getattr(operand, "not_", None) is not None:
                return self.evaluate_condition(Condition(not_=operand.not_), game_state, context)
            if getattr(operand, "max_", None) is not None:
                vals = [self.resolve_operand(x, game_state, context) for x in operand.max_]
                return max(vals)
            if getattr(operand, "min_", None) is not None:
                vals = [self.resolve_operand(x, game_state, context) for x in operand.min_]
                return min(vals)
            if getattr(operand, "sum_", None) is not None:
                vals = [self.resolve_operand(x, game_state, context) for x in operand.sum_]
                flat_vals: List[Any] = []
                for v in vals:
                    if isinstance(v, (list, tuple)):
                        flat_vals.extend(v)
                    else:
                        flat_vals.append(v)
                return sum(flat_vals)
            if getattr(operand, "count", None) is not None:
                collection = operand.count
                if isinstance(collection, list) and len(collection) == 1:
                    resolved = self.resolve_operand(collection[0], game_state, context)
                    return self._count_value(resolved)
                elif isinstance(collection, (list, tuple)):
                    return len(collection)
                else:
                    return self._count_value(collection)
            if getattr(operand, "rank_value", None) is not None:
                return self._op_rank_value(operand.rank_value[0], game_state, context)
            if getattr(operand, "top", None) is not None:
                container = self.resolve_operand(operand.top[0], game_state, context)
                card = None
                if hasattr(container, 'cards') and isinstance(container.cards, list):
                    card = container.cards[-1] if container.cards else None
                elif isinstance(container, list):
                    card = container[-1] if container else None
                return card
            if getattr(operand, "all_items", None) is not None:
                container = self.resolve_operand(operand.all_items[0], game_state, context)
                if hasattr(container, 'cards') and isinstance(container.cards, list):
                    return list(container.cards)
                return list(container) if isinstance(container, (list, tuple)) else []
            if getattr(operand, "add", None) is not None:
                vals = [self.resolve_operand(x, game_state, context) for x in operand.add]
                return sum(vals)
            if getattr(operand, "list_", None) is not None:
                return [self.resolve_operand(x, game_state, context) for x in operand.list_]
        return operand

    def execute_effect(self, effect_list: List[EffectAction], game_state: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """Executes a list of EffectAction models using the action registry. Supports FOR_EACH_PLAYER."""
        context = context or {}
        i = 0
        while i < len(effect_list):
            raw = effect_list[i]
            action_def = EffectAction.parse_obj(raw) if isinstance(raw, dict) else raw
            action_name = action_def.action

            # Special handling: FOR_EACH_PLAYER (only with inline 'do')
            if action_name == 'FOR_EACH_PLAYER':
                players_operand = getattr(action_def, 'players', None)
                players_list = self.resolve_operand(players_operand, game_state, context) if players_operand else getattr(game_state, 'players', [])
                # Convert to indices
                indices: List[int] = []
                for idx, p in enumerate(getattr(game_state, 'players', [])):
                    if not players_list:
                        indices.append(idx)
                    elif isinstance(players_list, list):
                        if p in players_list or (isinstance(players_list[0], int) and idx in players_list):
                            indices.append(idx)
                do_actions = getattr(action_def, 'do', None)
                if isinstance(do_actions, list) and do_actions:
                    for idx in indices:
                        local_ctx = dict(context)
                        local_ctx['$player'] = idx
                        self.execute_effect(do_actions, game_state, context=local_ctx)
                    i += 1
                    continue
                else:
                    # No inline 'do' provided -> reject (spec requires explicit do)
                    print("Action not implemented: FOR_EACH_PLAYER without 'do' is not supported.")
                    i += 1
                    continue

            # Normal action execution
            action_func = self.actions.get(action_name)

            if action_func:
                # Use by_alias=False so keys like 'from_' match function signatures
                raw_params = action_def.dict(exclude={"action"}, by_alias=False, exclude_none=True)
                params: Dict[str, Any] = {}
                for k, v in raw_params.items():
                    if isinstance(v, (dict, list)):
                        params[k] = self.resolve_operand(v, game_state, context)
                    else:
                        params[k] = v
                action_func(game_state, context=context, **params)
            else:
                print(f"Action not implemented: {action_name}")
            i += 1
