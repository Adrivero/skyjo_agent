from pathlib import Path

from agent.policy import HeuristicPolicy
from agent.ppo import RecurrentPolicy


class OpponentPolicy:
    """Load the recurrent Torch actor, falling back to a terminating heuristic."""

    def __init__(self, path, seed=None, device="auto", temperature=1.0):
        self.path = Path(path)
        self.seed = seed
        self.device = device
        self.temperature = temperature
        self.policy = None
        self.fallback = HeuristicPolicy(seed=seed)
        self.mode = "loading"
        self.warning = None

    @property
    def display_name(self):
        return "Learned MARL policy" if self.mode == "policy" else "Heuristic fallback"

    def load(self):
        try:
            if not self.path.is_file() or self.path.stat().st_size == 0:
                raise FileNotFoundError(self.path)
            self.policy = RecurrentPolicy.load(
                self.path,
                device=self.device,
                seed=self.seed,
                temperature=self.temperature,
            )
        except Exception as error:
            self.policy = None
            self.mode = "fallback"
            self.warning = (
                "Could not load the learned MARL policy; the opponent is using "
                f"the heuristic fallback ({type(error).__name__})."
            )
            return False
        self.mode = "policy"
        self.warning = None
        return True

    def reset(self, seed=None):
        self.fallback.reset(seed)
        if self.policy is not None:
            self.policy.reset(seed)

    def reset_round(self):
        self.fallback.reset_round()
        if self.policy is not None:
            self.policy.reset_round()

    def choose_action(self, observation, deterministic=False):
        if self.policy is not None:
            return self.policy.choose_action(observation, deterministic=deterministic)
        return self.fallback.choose_action(observation)
