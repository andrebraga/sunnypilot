"""
Copyright (c) 2021-, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from openpilot.cereal import log
from opendbc.car.interfaces import ACCEL_MAX

PARAM = "LimoModeCurves"
Personality = log.LongitudinalPersonality
PERSONALITIES = ("aggressive", "standard", "relaxed")  # as numbered in LongitudinalPersonality


@dataclass(frozen=True)
class Range:
  min: float
  max: float
  step: float

  def clip(self, value: float) -> float:
    return min(max(value, self.min), self.max)


RANGES = {
  "strength": Range(0.6, ACCEL_MAX, 0.05),  # m/s^2, the most it accelerates with
  "build": Range(0.2, 1.5, 0.05),  # s, when the acceleration builds the fastest
  "ease": Range(0.8, 4.0, 0.1),  # s, until a change is all the way through
}


@dataclass(frozen=True)
class LimoCurve:
  strength: float
  build: float
  ease: float

  def __post_init__(self):
    for name, r in RANGES.items():
      object.__setattr__(self, name, r.clip(float(getattr(self, name))))

  @property
  def ease_time(self) -> float:
    """It can't be all the way through before it's halfway there"""
    return max(self.ease, 2.0 * self.build)

  def towards(self, target: "LimoCurve", rates: "dict[str, float]", dt: float) -> "LimoCurve":
    """Changes must not step the output"""
    return replace(self, **{name: min(max(getattr(target, name), getattr(self, name) - rate * dt), getattr(self, name) + rate * dt)
                            for name, rate in rates.items()})


DEFAULTS = {
  "aggressive": LimoCurve(ACCEL_MAX, 0.2, 1.0),
  "standard": LimoCurve(ACCEL_MAX, 0.2, 1.4),
  "relaxed": LimoCurve(1.2, 1.0, 2.4),
}


def curves_from_param(value) -> dict[str, LimoCurve]:
  """Anything that doesn't make sense is the default"""
  curves = dict(DEFAULTS)
  if not isinstance(value, dict):
    return curves
  for name, default in DEFAULTS.items():
    stored = value.get(name)
    if not isinstance(stored, dict):
      continue
    values = {}
    for key in RANGES:
      v = stored.get(key)
      ok = isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
      values[key] = float(v) if ok else getattr(default, key)
    curves[name] = LimoCurve(**values)
  return curves


def curves_to_param(curves: dict[str, LimoCurve]) -> dict:
  return {name: {k: round(v, 3) for k, v in asdict(curve).items()} for name, curve in curves.items()}


def get_curve(curves: dict[str, LimoCurve], personality=Personality.standard) -> LimoCurve:
  if personality == Personality.aggressive:
    return curves["aggressive"]
  elif personality == Personality.standard:
    return curves["standard"]
  elif personality == Personality.relaxed:
    return curves["relaxed"]
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def ease_kernel(length: float, onset: float, dt: float) -> np.ndarray:
  """
  Unit area bump that peaks at onset. It has compact support and is infinitely differentiable, so anything filtered by
  it is too.
  """
  x = (np.arange(int(np.ceil(length / dt - 1e-9))) + 0.5) * dt / length
  x = np.minimum(x, 1.0 - 1e-6) ** (np.log(0.5) / np.log(onset / length))
  kernel = np.exp(-1.0 / (4.0 * x * (1.0 - x)))
  return kernel / kernel.sum()
