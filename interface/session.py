from agent.environment import (
    DRAW_ACTION,
    N_CARDS,
    TAKE_DISCARD_OFFSET,
    env,
)


class SkyjoSession:
    """UI-independent controller for one human-versus-policy series."""

    def __init__(self, opponent_policy, target_score=100):
        self.opponent_policy = opponent_policy
        self.target_score = target_score
        self.human_agent = "player_0"
        self.policy_agent = "player_1"
        self.environment = None
        self.match_history = []
        self.totals = {self.human_agent: 0, self.policy_agent: 0}
        self.match_number = 0
        self._match_recorded = False

    @property
    def match_active(self):
        return self.environment is not None and not self.match_over

    @property
    def match_over(self):
        return self.environment is not None and bool(self.environment.scores)

    @property
    def series_over(self):
        return any(score >= self.target_score for score in self.totals.values())

    @property
    def current_agent(self):
        if not self.match_active:
            return None
        return self.environment.agent_selection

    @property
    def human_turn(self):
        return self.current_agent == self.human_agent

    @property
    def turn_phase(self):
        if not self.match_active:
            return None
        return self.environment.turn_phase

    @property
    def last_match(self):
        return self.match_history[-1] if self.match_history else None

    @property
    def series_winner(self):
        if not self.series_over:
            return None
        human_total = self.totals[self.human_agent]
        policy_total = self.totals[self.policy_agent]
        if human_total == policy_total:
            return "tie"
        return self.human_agent if human_total < policy_total else self.policy_agent

    def start_match(self, human_positions, seed=None):
        if self.match_active:
            raise RuntimeError("The current match is still active.")
        if self.series_over:
            raise RuntimeError("Start a new series before playing another match.")

        positions = [tuple(position) for position in human_positions]
        if len(positions) != 2 or len(set(positions)) != 2:
            raise ValueError("Choose exactly two distinct opening positions.")

        self.environment = env()
        self.environment.reset(
            seed=seed,
            options={"initial_positions": {self.human_agent: positions}},
        )
        self.match_number = len(self.match_history) + 1
        self._match_recorded = False
        return self.current_agent

    def human_draw(self):
        self._require_human_phase("choose_source")
        self._step(DRAW_ACTION)

    def human_take_discard(self, card_index):
        self._require_human_phase("choose_source")
        self._step(TAKE_DISCARD_OFFSET + self._validate_index(card_index))

    def human_replace_with_drawn(self, card_index):
        self._require_human_phase("play_drawn")
        self._step(self._validate_index(card_index))

    def human_discard_and_reveal(self, card_index):
        self._require_human_phase("play_drawn")
        self._step(N_CARDS + self._validate_index(card_index))

    def step_policy(self):
        if not self.match_active or self.current_agent != self.policy_agent:
            raise RuntimeError("It is not the policy opponent's turn.")
        observation = self.environment.observe(self.policy_agent)
        action = self.opponent_policy.choose_action(observation)
        self._step(action)
        return action

    def action_is_legal(self, action):
        if not self.match_active:
            return False
        mask = self.environment.observe(self.current_agent)["action_mask"]
        return 0 <= action < len(mask) and bool(mask[action])

    def player(self, agent):
        if self.environment is None:
            return None
        return self.environment.players[agent]

    def new_series(self):
        self.environment = None
        self.match_history = []
        self.totals = {self.human_agent: 0, self.policy_agent: 0}
        self.match_number = 0
        self._match_recorded = False

    def _step(self, action):
        if not self.action_is_legal(action):
            raise ValueError(f"Action {action} is not legal in the current state.")
        self.environment.step(action)
        self._record_match_if_finished()

    def _record_match_if_finished(self):
        if not self.match_over or self._match_recorded:
            return

        scores = dict(self.environment.scores)
        raw_scores = {
            agent: self.environment.final_infos[agent]["raw_score"]
            for agent in self.environment.possible_agents
        }
        record = {
            "match": self.match_number,
            "scores": scores,
            "raw_scores": raw_scores,
            "closing_agent": self.environment.closing_agent,
        }
        self.match_history.append(record)
        for agent, score in scores.items():
            self.totals[agent] += score
        self._match_recorded = True

    def _require_human_phase(self, phase):
        if not self.match_active or not self.human_turn:
            raise RuntimeError("It is not the human player's turn.")
        if self.turn_phase != phase:
            raise RuntimeError(f"Expected turn phase {phase}, got {self.turn_phase}.")

    @staticmethod
    def _validate_index(card_index):
        card_index = int(card_index)
        if not 0 <= card_index < N_CARDS:
            raise ValueError(f"Card index must be between 0 and {N_CARDS - 1}.")
        return card_index
