"""Horizon scheduling.

Phase D adds a `RoundScheduler` ABC so the coordinator reads the
per-rollout horizon from a small composable object instead of grabbing
`request.horizon` directly. Two concretes:

  - FixedRoundsScheduler(horizon): the default; returns a constant.
    Equivalent to today's behavior.
  - ScalingRoundsScheduler(start, end, ramp_steps, mode): mirrors
    AgentGym-RL's ScalingInter-RL curriculum - horizon ramps from
    `start` to `end` over `ramp_steps` training steps. `mode="linear"`
    interpolates smoothly; `mode="step"` makes one discrete jump
    halfway through the ramp.

The scheduler is consulted at request time. Trainers that drive
training-step-aware curricula thread the current step into
`current_horizon(training_step=...)`; for one-off rollouts the default
`training_step=0` falls through to the scheduler's lower bound.

If a `RolloutRequest` carries an explicit `horizon`, it overrides the
scheduler. The scheduler only fills the gap when the request omits one.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


class RoundScheduler(ABC):
    """Resolve the per-rollout horizon (turn budget).

    Stateless from the scheduler's POV: callers thread in any
    training-step counter they're tracking; the scheduler decides what
    horizon to grant.
    """

    @abstractmethod
    def current_horizon(self, training_step: int = 0) -> int:
        ...


@dataclass(frozen=True)
class FixedRoundsScheduler(RoundScheduler):
    """Constant horizon. Equivalent to today's `request.horizon` default."""

    horizon: int = 6

    def current_horizon(self, training_step: int = 0) -> int:
        return self.horizon


@dataclass(frozen=True)
class ScalingRoundsScheduler(RoundScheduler):
    """Curriculum that ramps horizon from `start` to `end` over `ramp_steps`.

    Modes:
      - "linear": linear interpolation between (0, start) and
        (ramp_steps, end). Beyond ramp_steps the scheduler clamps to `end`.
      - "step":   `start` for the first half of the ramp, then `end`.
        Useful when intermediate horizon values would put the agent in
        a region the policy hasn't yet learned to handle.
    """

    start: int = 4
    end: int = 16
    ramp_steps: int = 100
    mode: Literal["linear", "step"] = "linear"

    def current_horizon(self, training_step: int = 0) -> int:
        if training_step <= 0:
            return self.start
        if training_step >= self.ramp_steps:
            return self.end
        if self.mode == "step":
            return self.start if training_step * 2 < self.ramp_steps else self.end
        # linear
        progress = training_step / self.ramp_steps
        return round(self.start + (self.end - self.start) * progress)
