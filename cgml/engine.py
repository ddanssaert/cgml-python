import re
from typing import Any, Dict, Callable, List, Union, Optional, Tuple

from .loader import Condition, Operand, EffectAction
from .state import card_owner, card_rank, find_card_zone

def _resolve_ref_placeholders(path: str, context: Dict[str, Any]) -> str:
    """Replace ref:<name> in path with value from context."""
    def repl(match):
        name = match.group(1)
        return str(context.get(name, ""))
    return re.sub(r"ref:([A-Za-z0-9_]+)", repl, path)

def resolve_path(obj: Any, path: str, context: Optional[Dict[str, Any]] = None) -> Any:
    """
    Selector resolver supporting CGML v1.3 anchors, selector filters, and basic path functions.
    """
    context = context or {}
    p = path.strip()
    if not p.startswith("$"):
        raise ValueError(f"Only $-rooted selector paths are supported (got: {path})")

    # 1. Anchor Replacement
    anchor_map = {
        "$currentPlayer": context.get("currentPlayer"),
        "$activeState": context.get("activeState"),
        "$currentPhase": context.get("currentPhase"),
        "$turnOrder": context.get("turnOrder"),
    }
    for anchor, val in anchor_map.items():
        if anchor in p and val is not None:
            if anchor == "$currentPlayer":
                # Replace with $.players[<index>]
                p = p.replace(anchor, f"$.players[{val}]")
            else:
                # Replace with literal value
                p = p.replace(anchor, str(val))

    # 2. Interpolate ref:<name> in path
    p = _resolve_ref_placeholders(p, context)

    # 3. Parse parts and handle selector filters
    #    (e.g. $.players[by_id=alice], $.players[current], $.players[opponent], $.players[team=red])
    # Basic path splitting
    # We'll add support for filters here after splitting into parts

    root = {
        "players": getattr(obj, "players", None),
        "zones": getattr(obj, "shared_zones", None),
    }
    current: Any = root
    parts: List[str] = []
    buf = ""
    i = 1
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

    # Walk through each part, handling filters
    for part in parts:
        # Detect [<filter>] or [<idx>]
        if '[' in part and part.endswith(']'):
            key = part[: part.index('[')]
            inside = part[part.index('[') + 1 : -1]
            # Support array access/filter
            if key == "players":
                plist = None
                # Support root access
                if isinstance(current, dict):
                    plist = current.get("players")
                elif hasattr(current, "players"):
                    plist = current.players
                else:
                    plist = current
                if inside == '*':
                    current = plist
                elif inside.isdigit():
                    current = [plist[int(inside)]]
                elif inside == "current":
                    val = context.get("currentPlayer", 0)
                    current = [plist[int(val)]] if plist else None
                elif inside == "opponent":
                    cur = int(context.get("currentPlayer", 0))
                    current = [p for idx, p in enumerate(plist) if idx != cur]
                elif inside.startswith("by_id="):
                    val = inside[6:]
                    current = [p for p in plist if str(getattr(p, "id", "")) == val]
                elif inside.startswith("team="):
                    val = inside[5:]
                    current = [p for p in plist if getattr(p, "team", None) == val]
                elif inside.startswith("$"):  # handle anchors like $player
                    anchor_name = inside[1:]  # e.g. 'player'
                    anchor_val = context.get(f"${anchor_name}")
                    if anchor_val is None:
                        raise ValueError(f"Selector uses anchor ${anchor_name} but context provides no value for it.")
                    current = [plist[int(anchor_val)]]
                else:
                    raise NotImplementedError(f"Selector filter not recognized: [{inside}] in path {path}")
                # Single result, skip down dictionary objects...
                if isinstance(current, list) and len(current) == 1:
                    current = current[0]
            else:
                # Descent for per-player zones, e.g. zones[deck]
                zimap = None
                if isinstance(current, dict):
                    zimap = current.get(key)
                elif hasattr(current, key):
                    zimap = getattr(current, key)
                else:
                    raise KeyError(f"Cannot access key '{key}' in {type(current)}")
                # Simplify: only array lookup, e.g. [hand] in zones
                if inside == '*':
                    current = list(zimap.values())
                else:
                    # Name lookups (zone_name etc)
                    current = zimap[inside] if isinstance(zimap, dict) else None
        else:
            key = part
            if isinstance(current, list):
                # Apply to all objects unless it's supposed to be a dict attr
                mapped: List[Any] = []
                for elem in current:
                    if isinstance(elem, dict):
                        mapped.append(elem.get(key))
                    else:
                        mapped.append(getattr(elem, key))
                current = mapped
                # flatten if result is singleton per list
                if len(current) == 1:
                    current = current[0]
            elif isinstance(current, dict):
                current = current.get(key)
            else:
                current = getattr(current, key)
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

    def _is_rank_literal(self, value: Any, cgml_def: Any) -> bool:
        """
        Returns True if the value is a rank (string/int) in any deck_type's rank_hierarchy.
        Ignores ints, floats, and lists (should only be called for rank comparison).
        """
        if isinstance(value, (int, float)):
            return False
        if isinstance(value, list):
            return False
        if not (cgml_def and cgml_def.components and cgml_def.components.component_types):
            return False
        deck_types = cgml_def.components.component_types.get('deck_types', {})
        for deck_name in deck_types:
            hierarchy = [str(x) for x in deck_types[deck_name].rank_hierarchy]
            if str(value) in hierarchy:
                return True
        return False

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
        if hasattr(value, 'cards') and isinstance(getattr(value, 'cards'), list):
            return len(getattr(value, 'cards'))
        if hasattr(value, 'card_count'):
            try:
                return int(getattr(value, 'card_count'))
            except Exception:
                pass
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
        """
        Recursively evaluates a Condition (pydantic model or dict node).
        Enforces explicit use of rank_value in all rank comparisons.
        """
        context = context or {}
        if not isinstance(cond, (Condition, dict)):
            return bool(cond)
        if isinstance(cond, dict):
            cond = Condition.parse_obj(cond)
        cgml_def = getattr(game_state, "cgml_definition", None)

        def _raise_rank_error(op_name: str) -> None:
            raise ValueError(
                f"CGML spec: '{op_name}' between ranks requires explicit use of 'rank_value' operator. "
                f"No implicit string-based rank comparison is permitted (§12.5, §15, §19)."
            )

        if cond.isEqual is not None:
            left = self.resolve_operand(cond.isEqual[0], game_state, context)
            right = self.resolve_operand(cond.isEqual[1], game_state, context)
            lv = cond.isEqual[0]
            rv = cond.isEqual[1]
            lv_is_rank_val = isinstance(lv, dict) and "rank_value" in lv
            rv_is_rank_val = isinstance(rv, dict) and "rank_value" in rv
            rank_comparison = self._is_rank_literal(left, cgml_def) or self._is_rank_literal(right, cgml_def)
            if rank_comparison and not (lv_is_rank_val and rv_is_rank_val):
                print(f"DEBUG RANK_ENFORCE: left={left} ({type(left)}), right={right} ({type(right)}), "
                      f"lv={lv}, rv={rv}, rank_comparison={rank_comparison}")
                _raise_rank_error("isEqual")
            return left == right
        if cond.isGreaterThan is not None:
            left = self.resolve_operand(cond.isGreaterThan[0], game_state, context)
            right = self.resolve_operand(cond.isGreaterThan[1], game_state, context)
            lv = cond.isGreaterThan[0]
            rv = cond.isGreaterThan[1]
            lv_is_rank_val = isinstance(lv, dict) and "rank_value" in lv
            rv_is_rank_val = isinstance(rv, dict) and "rank_value" in rv
            rank_comparison = self._is_rank_literal(left, cgml_def) or self._is_rank_literal(right, cgml_def)
            if rank_comparison and not (lv_is_rank_val and rv_is_rank_val):
                print(f"DEBUG RANK_ENFORCE: left={left} ({type(left)}), right={right} ({type(right)}), "
                      f"lv={lv}, rv={rv}, rank_comparison={rank_comparison}")
                _raise_rank_error("isGreaterThan")
            return left > right
        if cond.isLessThan is not None:
            left = self.resolve_operand(cond.isLessThan[0], game_state, context)
            right = self.resolve_operand(cond.isLessThan[1], game_state, context)
            lv = cond.isLessThan[0]
            rv = cond.isLessThan[1]
            lv_is_rank_val = isinstance(lv, dict) and "rank_value" in lv
            rv_is_rank_val = isinstance(rv, dict) and "rank_value" in rv
            rank_comparison = self._is_rank_literal(left, cgml_def) or self._is_rank_literal(right, cgml_def)
            if rank_comparison and not (lv_is_rank_val and rv_is_rank_val):
                print(f"DEBUG RANK_ENFORCE: left={left} ({type(left)}), right={right} ({type(right)}), "
                      f"lv={lv}, rv={rv}, rank_comparison={rank_comparison}")
                _raise_rank_error("isLessThan")
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
        if hasattr(cond, "distinct") and cond.distinct is not None:
            items = self.resolve_operand(cond.distinct[0], game_state, context)
            try:
                items = list(items)
            except Exception:
                pass
            result, seen = [], set()
            for it in items:
                val = it
                # Scalars/dicts? Use id or as-is
                if isinstance(it, dict) and "id" in it:
                    val = it["id"]
                elif hasattr(it, "id"):
                    val = it.id
                if val not in seen:
                    seen.add(val)
                    result.append(it)
            return result
        if hasattr(cond, "group_by") and cond.group_by is not None:
            list_items = self.resolve_operand(cond.group_by[0], game_state, context)
            key_expr = cond.group_by[1]
            grouped = {}
            for item in list_items:
                group_ctx = dict(context or {})
                group_ctx["item"] = item
                key = self.resolve_operand(key_expr, game_state, group_ctx)
                grouped.setdefault(key, []).append(item)
            return grouped
        return False

    def resolve_operand(
        self,
        operand: Union[Operand, Dict, Any],
        game_state: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Resolves an operand node: can be Operand model, dict, or value.
        """
        context = context or {}

        if isinstance(operand, dict) and len(operand) == 1:
            op, arg = list(operand.items())[0]
            if op == "top":
                x = self.resolve_operand(arg[0], game_state, context)
                if hasattr(x, "cards"):
                    return x.top_card if hasattr(x, "top_card") else (x.cards[-1] if x.cards else None)
                elif isinstance(x, list):
                    return x[-1] if x else None
                return x
            elif op == "bottom":
                x = self.resolve_operand(arg[0], game_state, context)
                if hasattr(x, "bottom_card"):
                    return x.bottom_card
                elif hasattr(x, "cards"):
                    return x.cards[0] if x.cards else None
                elif isinstance(x, list):
                    return x[0] if x else None
                return x
            elif op == "all":
                x = self.resolve_operand(arg[0], game_state, context)
                # Accept both Zone and list
                if hasattr(x, "all_cards"):
                    return x.all_cards()
                elif hasattr(x, "cards"):
                    return x.cards[:]
                return list(x) if isinstance(x, (list, tuple)) else []
            elif op == "count":
                x = self.resolve_operand(arg[0], game_state, context)
                if hasattr(x, "card_count"):
                    return x.card_count
                try:
                    return len(x)
                except Exception:
                    return 1 if x is not None else 0
            elif op == "owner":
                y = self.resolve_operand(arg[0], game_state, context)
                return card_owner(y, game_state) if y else None
            elif op == "rank":
                y = self.resolve_operand(arg[0], game_state, context)
                return card_rank(y)
            elif op == "distinct":
                return self.evaluate_condition({"distinct": arg}, game_state, context)
            elif op == "group_by":
                return self.evaluate_condition({"group_by": arg}, game_state, context)
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

    def _resolve_action_params(self, action_def: EffectAction, game_state: Any, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        action_name = action_def.action
        raw_params = action_def.dict(exclude={"action"}, by_alias=False, exclude_none=True)
        params: Dict[str, Any] = {}
        for k, v in raw_params.items():
            if isinstance(v, (dict, list)):
                params[k] = self.resolve_operand(v, game_state, context)
            else:
                params[k] = v
        return action_name, params

    def _collect_player_indices(self, players_operand: Any, game_state: Any, context: Dict[str, Any]) -> List[int]:
        indices: List[int] = []
        if players_operand is None:
            return list(range(len(getattr(game_state, 'players', []))))
        resolved = self.resolve_operand(players_operand, game_state, context)
        all_players = list(getattr(game_state, 'players', []))
        if isinstance(resolved, list) and resolved and isinstance(resolved[0], int):
            for i in resolved:
                if 0 <= int(i) < len(all_players):
                    indices.append(int(i))
            return indices
        if not resolved:
            return list(range(len(all_players)))
        for idx, p in enumerate(all_players):
            if p in resolved:
                indices.append(idx)
        return indices

    def execute_effect(self, effect_list: List[EffectAction], game_state: Any, context: Optional[Dict[str, Any]] = None) -> None:
        context = context or {}
        i = 0
        while i < len(effect_list):
            raw = effect_list[i]
            action_def = EffectAction.parse_obj(raw) if isinstance(raw, dict) else raw
            action_name = action_def.action

            if action_name == 'FOR_EACH_PLAYER':
                players_operand = getattr(action_def, 'players', None)
                indices = self._collect_player_indices(players_operand, game_state, context)
                order = (getattr(action_def, 'order', None) or 'clockwise').lower()
                if order == 'counterclockwise':
                    indices = list(reversed(indices))
                do_actions = getattr(action_def, 'do', None)
                if not (isinstance(do_actions, list) and do_actions):
                    print("Action not implemented: FOR_EACH_PLAYER without 'do' is not supported.")
                    i += 1
                    continue

                if order == 'simultaneous':
                    for step_raw in do_actions:
                        step_def = EffectAction.parse_obj(step_raw) if isinstance(step_raw, dict) else step_raw
                        step_name = step_def.action
                        action_func = self.actions.get(step_name)
                        if not action_func:
                            print(f"Action not implemented: {step_name}")
                            continue
                        batch: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
                        for idx_val in indices:
                            local_ctx = dict(context)
                            local_ctx['$player'] = idx_val
                            step_def_local = EffectAction.parse_obj(step_raw) if isinstance(step_raw, dict) else step_raw
                            _, params = self._resolve_action_params(step_def_local, game_state, local_ctx)
                            batch.append((params, local_ctx))
                        for params, local_ctx in batch:
                            action_func(game_state, context=local_ctx, **params)
                    i += 1
                    continue
                else:
                    for idx_val in indices:
                        local_ctx = dict(context)
                        local_ctx['$player'] = idx_val
                        self.execute_effect(do_actions, game_state, context=local_ctx)
                    i += 1
                    continue

            if action_name == 'PARALLEL':
                branches_raw = getattr(action_def, 'do', None)
                wait_mode = (getattr(action_def, 'wait', None) or 'all').lower()
                if not isinstance(branches_raw, list) or not branches_raw:
                    i += 1
                    continue
                branches: List[List[EffectAction]] = []
                for br in branches_raw:
                    if isinstance(br, list):
                        branches.append([EffectAction.parse_obj(x) if isinstance(x, dict) else x for x in br])
                    else:
                        branches.append([EffectAction.parse_obj(br) if isinstance(br, dict) else br])
                if wait_mode == 'all':
                    for branch in branches:
                        self.execute_effect(branch, game_state, context=dict(context))
                else:
                    for branch in branches:
                        self.execute_effect(branch, game_state, context=dict(context))
                i += 1
                continue

            if action_name == 'IF':
                cond_node = getattr(action_def, 'condition', None)
                then_actions = getattr(action_def, 'then', None)
                else_actions = getattr(action_def, 'else', None)
                cond_ok = self.evaluate_condition(cond_node, game_state, context) if cond_node is not None else False
                if cond_ok and isinstance(then_actions, list):
                    self.execute_effect(then_actions, game_state, context=dict(context))
                elif (not cond_ok) and isinstance(else_actions, list):
                    self.execute_effect(else_actions, game_state, context=dict(context))
                i += 1
                continue

            action_func = self.actions.get(action_name)

            if action_func:
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
