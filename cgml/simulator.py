from typing import Any, Dict, List, Optional
import random
import logging
from .engine import RulesEngine
from .state import (
    GameState,
    build_game_state_from_cgml,
    run_setup_phase,
    find_zone,
    move_cards,
    find_card_zone,
    move_all_cards,
    shuffle_zone,
)
from .loader import load_cgml_file

# Set up debug logger
logger = logging.getLogger("simulator")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(ch)

# --- Action Registry Setup ---

def move_action(game_state: GameState, from_: Any, to: Any, count: Optional[int] = 1, context=None, **kwargs) -> None:
    """MOVE cards between zones.

    - from_: Zone, Card, or path-like resolved object (may resolve to None if empty via top()).
    - to: Zone or path-like resolved object.
    - count: number of cards (defaults to 1). Ignored when from_ is a Card.
    """
    # Graceful no-op if there is nothing to move or destination missing
    if from_ is None or to is None:
        return

    # Normalize count
    try:
        cnt = int(count) if count is not None else 1
    except Exception:
        cnt = 1

    # Resolve destination zone first
    to_zone = to if hasattr(to, 'cards') else find_zone(game_state, to)

    # If from_ is a Card, move exactly that card
    if hasattr(from_, 'id') and hasattr(from_, 'properties'):
        from_zone = find_card_zone(game_state, from_)
        if from_zone is None:
            return  # nothing to move; fail-safe
        for idx, c in enumerate(from_zone.cards):
            if c is from_ or getattr(c, 'id', None) == getattr(from_, 'id', None):
                card_obj = from_zone.cards.pop(idx)
                to_zone.cards.append(card_obj)
                break
        return

    # Otherwise treat as zone reference
    from_zone = from_ if hasattr(from_, 'cards') else find_zone(game_state, from_)
    move_cards(from_zone, to_zone, cnt)


def move_all_action(game_state: GameState, from_: Any = None, to: Any = None, context=None, **kwargs) -> None:
    # Graceful no-op if params are missing
    if from_ is None or to is None:
        return
    from_zone = from_ if hasattr(from_, 'cards') else find_zone(game_state, from_)
    to_zone = to if hasattr(to, 'cards') else find_zone(game_state, to)
    move_all_cards(from_zone, to_zone)


def deal_action(game_state: GameState, from_: Any, to: Any, count: Optional[int] = 1, context=None, **kwargs) -> None:
    """DEAL from a source zone to a single target zone (runtime).

    Semantics: identical to moving N cards from the source to the target zone.
    """
    from .state import find_zone, move_cards

    if from_ is None or to is None:
        return

    try:
        cnt = int(count) if count is not None else 1
    except Exception:
        cnt = 1

    from_zone = from_ if hasattr(from_, 'cards') else find_zone(game_state, from_)
    to_zone = to if hasattr(to, 'cards') else find_zone(game_state, to)
    move_cards(from_zone, to_zone, cnt)


def deal_round_robin_action(game_state: GameState, from_: Any, to: Any, count: Optional[int] = 1, order: Optional[str] = None, context=None, **kwargs) -> None:
    """DEAL_ROUND_ROBIN from a source zone to a list of target zones (runtime).

    - from_: source Zone or path
    - to: list of Zones (e.g., resolved from $.players[*].zones.hand) or a selector path
    - count: number of rounds (each round gives one card to each target in order)
    - order: 'clockwise' (default) or 'counterclockwise'
    """
    if from_ is None or to is None:
        return

    try:
        rounds = int(count) if count is not None else 1
    except Exception:
        rounds = 1

    from_zone = from_ if hasattr(from_, 'cards') else find_zone(game_state, from_)

    # Normalize targets to a list of Zone objects
    if isinstance(to, list):
        targets = [z for z in to if hasattr(z, 'cards')]
    else:
        # Expect a selector yielding list of zones
        maybe = to if hasattr(to, 'cards') else find_zone(game_state, to)
        targets = [maybe] if hasattr(maybe, 'cards') else []

    if not targets:
        return

    if (order or '').lower() == 'counterclockwise':
        targets = list(reversed(targets))

    for _ in range(rounds):
        for tz in targets:
            if from_zone.cards:
                tz.cards.append(from_zone.cards.pop())


def set_state_action(game_state: GameState, state: str, context=None, **kwargs) -> None:
    """Set the current FSM state."""
    game_state.current_state = state


def shuffle_action(game_state: GameState, target: Any, context=None, **kwargs) -> None:
    zone = target if hasattr(target, 'cards') else find_zone(game_state, target)
    shuffle_zone(zone)


# Extendable action registry (subset needed for war.yml)
ACTION_REGISTRY = {
    "MOVE": move_action,
    "MOVE_ALL": move_all_action,
    "DEAL": deal_action,
    "DEAL_ROUND_ROBIN": deal_round_robin_action,
    "SET_STATE": set_state_action,
    "SHUFFLE": shuffle_action,
}


class GameSimulator:
    def __init__(self, cgml_definition: Any, player_count: int):
        """
        Initialize simulator with rules engine and built game state.
        Ensures RNG seeding occurs before any state construction or setup so that
        SHUFFLE and any random choices are deterministic when configured.
        """
        self.cgml_definition = cgml_definition

        # RNG seeding based on meta.rng (seed before building state/setup)
        rng_cfg = getattr(self.cgml_definition.meta, 'rng', None)
        if rng_cfg and getattr(rng_cfg, 'deterministic', False):
            seed_val = getattr(rng_cfg, 'seed', None)
            random.seed(seed_val)
            logger.debug(f"Deterministic RNG enabled. Seed={seed_val}")

        self.rules_engine = RulesEngine(ACTION_REGISTRY)
        self.player_count = player_count
        self.game_state = self._initialize_state(cgml_definition)
        self.flow = cgml_definition.flow

        self.game_state.current_state = self.flow.initial_state
        self.current_player_idx = 0
        self.phase_idx = 0

    def _initialize_state(self, cgml_def: Any) -> GameState:
        state = build_game_state_from_cgml(cgml_def, self.player_count)
        run_setup_phase(state)
        return state

    def get_legal_actions(self, player_id: int) -> List[Dict]:
        """Collect rules with triggers matching the current phase."""
        legal_actions: List[Dict] = []
        current_phase = self._current_phase()
        for rule in self.cgml_definition.rules:
            if rule.trigger == f"on.phase.{current_phase}":
                if not rule.condition or self.rules_engine.evaluate_condition(rule.condition, self.game_state):
                    if rule.effect:
                        legal_actions.append({
                            "rule_id": rule.id,
                            "effect": rule.effect,
                        })
        return legal_actions

    def _get_phases_for_state(self, state_name: str) -> List[str]:
        state_def = self.flow.states.get(state_name)
        if state_def and state_def.phases:
            return state_def.phases
        return []

    def _current_phase(self) -> Optional[str]:
        phases = self._get_phases_for_state(self.game_state.current_state)
        if 0 <= self.phase_idx < len(phases):
            return phases[self.phase_idx]
        return None

    def _check_state_transitions(self) -> bool:
        transitions = self.flow.transitions or []
        for t in transitions:
            if t.from_ == self.game_state.current_state:
                if not t.condition or self.rules_engine.evaluate_condition(t.condition, self.game_state):
                    self.game_state.current_state = t.to
                    self.phase_idx = 0
                    return True
        return False

    def run(self) -> None:
        """Advance phases/turns until GameOver or actions exhausted."""
        while True:
            logger.debug(
                f"Sim state: {self.game_state.current_state}, phase: {self._current_phase()}, player: {self.current_player_idx}"
            )
            card_counts = ',\t'.join([
                f'{z.name}: {len(z.cards)}' for z in self.game_state.shared_zones.values()
            ] + [
                f'{p.name} {z.name}: {len(z.cards)}' for p in self.game_state.players for z in p.zones.values()
            ])
            logger.info(f"Zone card counts: {card_counts}")

            if self._is_game_over():
                logger.debug("Game over detected by simulator.")
                print("Game over!")
                break

            player_id = self.current_player_idx
            legal = self.get_legal_actions(player_id)

            if not legal:
                logger.debug(
                    f"No legal actions for player {player_id} in phase {self._current_phase()}. Attempting to advance phase."
                )
                # Before advancing, check if a transition (like GameOver) should fire at this checkpoint
                if self._check_state_transitions():
                    logger.debug(
                        f"State changed by flow.transition to {self.game_state.current_state}; resetting phase index."
                    )
                    if self._is_game_over():
                        logger.debug(
                            f"Sim state: {self.game_state.current_state}, phase: {self._current_phase()}, player: {self.current_player_idx}"
                        )
                        card_counts = ',\t'.join(
                            [f'{z.name}: {len(z.cards)}' for z in self.game_state.shared_zones.values()] + [
                                f'{p.name} {z.name}: {len(z.cards)}' for p in self.game_state.players for z in p.zones.values()]
                        )
                        logger.info(f"Zone card counts: {card_counts}")
                        print("Game over!")
                        break
                    continue

                if not self._advance_phase():
                    logger.debug("No more phases to advance. Breaking simulation loop.")
                    break
                continue

            prev_state_name = self.game_state.current_state

            # Choose next legal action deterministically if rng.deterministic, otherwise random
            rng_cfg = getattr(self.cgml_definition.meta, 'rng', None)
            if rng_cfg and getattr(rng_cfg, 'deterministic', False):
                selected_action = legal[0]  # deterministic pick: first legal rule
                logger.debug("Deterministic action selection: picking first legal rule.")
            else:
                selected_action = random.choice(legal)
                logger.debug("Random action selection:")

            logger.debug(
                f"Executing action (rule_id={selected_action['rule_id']}) with effect: {selected_action['effect']}"
            )
            self.rules_engine.execute_effect(selected_action['effect'], self.game_state)

            if self.game_state.current_state != prev_state_name:
                logger.debug(f"State changed {prev_state_name} -> {self.game_state.current_state}; resetting phase index.")
                self.phase_idx = 0
                continue

            if self._check_state_transitions():
                logger.debug(
                    f"State changed by flow.transition to {self.game_state.current_state}; resetting phase index."
                )
                if self._is_game_over():
                    logger.debug(
                        f"Sim state: {self.game_state.current_state}, phase: {self._current_phase()}, player: {self.current_player_idx}"
                    )
                    card_counts = ',\t'.join(
                        [f'{z.name}: {len(z.cards)}' for z in self.game_state.shared_zones.values()] + [
                            f'{p.name} {z.name}: {len(z.cards)}' for p in self.game_state.players for z in p.zones.values()]
                    )
                    logger.info(f"Zone card counts: {card_counts}")
                    print("Game over!")
                    break
                continue

            self._advance_phase()

    def _is_game_over(self) -> bool:
        return self.game_state.current_state == "GameOver"

    def _advance_turn(self) -> None:
        num_players = len(self.game_state.players)
        old_idx = self.current_player_idx
        self.current_player_idx = (self.current_player_idx + 1) % num_players
        logger.debug(f"Advanced turn: player {old_idx} -> {self.current_player_idx}")

    def _advance_phase(self) -> bool:
        phases = self._get_phases_for_state(self.game_state.current_state)
        if not phases:
            logger.debug(f"No phases in state {self.game_state.current_state}.")
            return False
        old_idx = self.phase_idx
        self.phase_idx += 1
        if self.phase_idx >= len(phases):
            self.phase_idx = 0
            self._advance_turn()
            logger.debug(
                f"Phase wrapped for state {self.game_state.current_state}. Advancing to new turn."
            )
            return False
        logger.debug(
            f"Advanced phase {old_idx} -> {self.phase_idx} ({phases[self.phase_idx]}) in state {self.game_state.current_state}."
        )
        return True


# --- Usage Example ---
if __name__ == "__main__":
    cgml = load_cgml_file("../cgml-specification/games/war.yml")  # or any other CGML .yml game file
    simulator = GameSimulator(cgml, player_count=2)
    simulator.run()
