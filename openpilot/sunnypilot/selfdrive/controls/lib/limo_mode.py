"""
Copyright (c) 2021-, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from functools import wraps

import numpy as np

from openpilot.cereal import custom, log
from opendbc.car.interfaces import ACCEL_MAX, ACCEL_MIN
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD

LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource

# Cruise goes through a window that responds as quickly as without one, and then takes its time to get all the way there
EASE_TIME = (1.0, 1.4, 1.8)  # aggressive, standard, relaxed, s
EASE_TIME_MAX = max(EASE_TIME)
EASE_TIME_RATE = 0.3  # s/s, changes must not step the output
EASE_ONSET = 0.2  # s, where the window peaks
# Everything goes through one that's just long enough to round off the corners
BLEND_TIME = 0.2  # s
BLEND_ONSET = 0.08  # s
LEAD_LOOKAHEAD = 0.1  # s, on top of the window's delay. following a lead responds as quickly as without smoothing
CRUISE_HEADROOM = 1.0  # m/s^2, how far cruise stays above the lead or (boosted) model target that limits it
TARGET_STEP = 2.0  # m/s^2, bounds the jerk of big changes

CRUISE_TAU = 0.5  # s, time constant of the last bit of a speed change
VISION_TIME = 4.0  # s, it comes and goes in a curve, stay out of it for this long after
CRUISE_DECEL = 0.5  # m/s^2, never harder than without Limo mode, the set speed and the map can be wrong
APPROACH_DECEL = 0.45  # m/s^2, what to plan with. slowing for something at a known distance starts earlier instead
APPROACH_DECEL_MIN = 0.2  # m/s^2, the car doesn't slow down as it's told to
ARRIVAL_TAUS = 1.5  # cruise time constants until a slowdown counts as arrived

# Smoothing must never make the car brake less than the lead and model ask for. How far above their targets the output
# can be while a change blends in, none of it is left once they ask for real braking
BLEND_MARGIN_BP = [-0.5, 0.0, 0.5]  # m/s^2, lowest of the lead and model targets
BLEND_MARGIN_V = [0.0, 0.2, 0.5]  # m/s^2
# The lead plan is already jerk limited, stay closer to it
LEAD_MARGIN_V = [0.0, 0.1, 0.2]  # m/s^2
CRAWL_SPEED = (4.0, 2.0)  # m/s, stick to the lead and model targets to pull away and stop exactly like without smoothing
MODEL_LOOKAHEAD_DECEL = 0.5  # m/s^2, how much earlier braking to take from the model's plan

# Don't keep rolling towards a close lead with the brakes barely on
CREEP_DISTANCE = (4.5, 3.5)  # m, closer than where a stop should end
CREEP_ACCEL = (0.1, -0.05)  # m/s^2, when the lead plan doesn't want to go
CREEP_SPEED = (0.02, 0.2)  # m/s
CREEP_DECEL = 0.4  # m/s^2

# The part of the car's response that isn't from the commands. Cars that hold the commanded accel on hills have little
DISTURBANCE_TAU = 1.5  # s
DISTURBANCE_MAX = 0.4  # m/s^2
DISTURBANCE_MIN_SPEED = 2.5  # m/s


def get_ease_time(personality=log.LongitudinalPersonality.standard) -> float:
  if personality == log.LongitudinalPersonality.relaxed:
    return EASE_TIME[2]
  elif personality == log.LongitudinalPersonality.standard:
    return EASE_TIME[1]
  elif personality == log.LongitudinalPersonality.aggressive:
    return EASE_TIME[0]
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def smoothstep(x: float, start: float, end: float) -> float:
  t = float(np.clip((x - start) / (end - start), 0.0, 1.0))
  return t * t * (3.0 - 2.0 * t)


def ease_kernel(length: float, onset: float, dt: float) -> np.ndarray:
  """
  Unit area bump that peaks at onset. It has compact support and is infinitely differentiable, so anything filtered by
  it is too.
  """
  x = (np.arange(int(np.ceil(length / dt - 1e-9))) + 0.5) * dt / length
  x = np.minimum(x, 1.0 - 1e-6) ** (np.log(0.5) / np.log(onset / length))
  kernel = np.exp(-1.0 / (4.0 * x * (1.0 - x)))
  return kernel / kernel.sum()


def fail_safe(default):
  """Limo mode must not take the planner down, or get a second chance to"""
  def decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
      if self.failed:
        return default
      try:
        return func(self, *args, **kwargs)
      except Exception:
        cloudlog.exception(f"limo mode failed in {func.__name__}, off until restart")
        self.failed = True
        self.enabled = False
        return default
    return wrapper
  return decorator


class Window:
  """Filter that knows what is still on its way through it"""
  def __init__(self, length_max: float, dt: float):
    self.dt = dt
    self.n = int(round(length_max / dt))
    self.history = np.zeros(self.n)  # newest first

  def set_kernel(self, kernel: np.ndarray) -> None:
    self.kernel = np.zeros(self.n)
    self.kernel[:len(kernel)] = kernel
    self.length = len(kernel) * self.dt
    self.delay = float(self.kernel @ np.arange(self.n)) * self.dt
    # share of each past input that has not come out yet, and the sum of that over time
    remaining = np.cumsum(self.kernel[::-1])[::-1]
    self.pending = np.append(remaining[1:], 0.0)
    self.pending_moment = np.cumsum(self.pending[::-1])[::-1] * self.dt

  def reset(self, value: float) -> None:
    self.history[:] = value

  def update(self, value: float) -> float:
    self.history = np.roll(self.history, 1)
    self.history[0] = value
    return self.value

  @property
  def value(self) -> float:
    return float(self.history @ self.kernel)


class LimoMode:
  """
  Smooths the longitudinal target without arriving late.

  Cruise works with the speed the car is already committed to: what it has now plus all the acceleration still in the
  windows and in the actuators. The sum of the acceleration is the same as without smoothing, so the car still ends up
  at the same speed and place. Lead and model targets only go through the short window, and their plans are read ahead
  by its delay.

  What it can't do:
  - brake less than the lead and model targets, other than by the blend margins
  - brake more than CRUISE_DECEL on its own
  - accelerate past ACCEL_MAX or the set speed
  """
  def __init__(self, CP, dt: float = DT_MDL):
    self.params = Params()
    self.dt = dt
    self.openpilot_longitudinal = CP.openpilotLongitudinalControl
    self.actuator_delay = CP.longitudinalActuatorDelay
    self.frame = -1
    self.enabled = False
    self.active = False
    self.failed = False

    self.ease_time = get_ease_time()
    self.cruise = Window(EASE_TIME_MAX, dt)
    self.cruise.set_kernel(ease_kernel(self.ease_time, EASE_ONSET, dt))
    self.blend = Window(BLEND_TIME, dt)
    self.blend.set_kernel(ease_kernel(BLEND_TIME, BLEND_ONSET, dt))
    self.lookahead = self.blend.delay + LEAD_LOOKAHEAD  # how much further ahead to read the lead and model plans
    self.n_delay = int(round(self.actuator_delay / dt))
    self.outputs = np.zeros(max(self.n_delay, 1))  # newest first
    self.a_plan = 0.0
    self.output_last = 0.0
    self.v_ego = 0.0
    self.disturbance = 0.0
    self.vision_time = 0.0

  @fail_safe(None)
  def update_params(self) -> None:
    self.frame += 1
    if self.frame % int(PARAMS_UPDATE_PERIOD / self.dt) == 0:
      self.enabled = self.openpilot_longitudinal and self.params.get_bool("LimoMode")

  @fail_safe(None)
  def reset(self, accel: float) -> None:
    self.cruise.reset(accel)
    self.blend.reset(accel)
    self.outputs[:] = accel
    self.a_plan = accel
    self.disturbance = 0.0

  @fail_safe(None)
  def start_frame(self, v_ego: float, a_ego: float, personality, active: bool) -> None:
    self.v_ego = v_ego
    self.active = active
    ease_time = get_ease_time(personality)
    if ease_time != self.ease_time:
      self.ease_time = float(np.clip(ease_time, self.ease_time - EASE_TIME_RATE * self.dt, self.ease_time + EASE_TIME_RATE * self.dt))
      self.cruise.set_kernel(ease_kernel(self.ease_time, EASE_ONSET, self.dt))
    if active and v_ego > DISTURBANCE_MIN_SPEED:
      disturbance = float(np.clip(a_ego - self.outputs[-1], -DISTURBANCE_MAX, DISTURBANCE_MAX))
      self.disturbance += (disturbance - self.disturbance) * self.dt / DISTURBANCE_TAU

  def _pending(self, window: Window) -> float:
    return float((window.history + self.disturbance) @ window.pending) * self.dt

  @property
  def _pending_actuators(self) -> float:
    return float((self.outputs[:self.n_delay] + self.disturbance).sum()) * self.dt

  @property
  def committed_speed(self) -> float:
    """Speed after everything that's already on its way has happened"""
    return self.v_ego + self._pending(self.cruise) + self._pending(self.blend) + self._pending_actuators

  @fail_safe(0.0)
  def approach_distance(self, v_target: float) -> float:
    """Distance to slow down to v_target, given everything the car is already committed to"""
    dv = self.committed_speed - v_target
    if not dv > 0.0:
      return 0.0

    # constant decel followed by an exponential tail
    decel = max(APPROACH_DECEL - max(self.disturbance, 0.0), APPROACH_DECEL_MIN)
    tail = min(dv, decel * CRUISE_TAU)
    t_constant = (dv - tail) / decel
    moment = (dv - tail) * t_constant / 2.0 + tail * (t_constant + CRUISE_TAU)

    t_end = t_constant + ARRIVAL_TAUS * CRUISE_TAU + self.ease_time + BLEND_TIME + self.actuator_delay
    distance = self.v_ego * t_end
    distance -= dv * (t_end - self.cruise.delay - self.blend.delay - self.actuator_delay) - moment
    # the pending changes happen on their own schedule
    for window, delay in ((self.cruise, self.blend.delay + self.actuator_delay), (self.blend, self.actuator_delay)):
      distance += self._pending(window) * (t_end - delay) - float((window.history + self.disturbance) @ window.pending_moment) * self.dt
    distance += self._pending_actuators * (t_end - self.actuator_delay / 2.0)
    return max(distance, 0.0)

  @fail_safe(None)
  def cruise_target(self, v_cruise: float, source) -> float | None:
    """Unsmoothed cruise target, None when Limo mode has to stay out of it"""
    # the vision controller's target speed moves with the speed of the car and is tuned for the planner's own response
    self.vision_time = VISION_TIME if source == LongitudinalPlanSource.sccVision else max(self.vision_time - self.dt, 0.0)
    if self.vision_time > 0.0:
      return None

    accel = (v_cruise - self.committed_speed) / CRUISE_TAU
    target = float(np.clip(accel - self.disturbance, -CRUISE_DECEL, ACCEL_MAX))
    return target if np.isfinite(target) else None

  @staticmethod
  def model_accel(a_target: float, a_now: float, a_ahead: float) -> float:
    """The model's target, braking earlier when its own plan brakes more later"""
    return a_target + float(np.clip(min(a_ahead, 0.0) - min(a_now, 0.0), -MODEL_LOOKAHEAD_DECEL, 0.0))

  def cruise_accel(self, target: float, a_limit: float) -> float:
    """
    Smoothed cruise target. a_limit is the lowest of the lead and model targets, cruise follows it so that it doesn't
    have to be smoothed all over again when it's not limited anymore.
    """
    last = self.cruise.value
    target = min(target, a_limit + CRUISE_HEADROOM)
    return self.cruise.update(float(np.clip(target, last - TARGET_STEP, last + TARGET_STEP)))

  @staticmethod
  def _limit(output: float, a_now: float, margin: list[float], follow: float = 0.0) -> float:
    return min(output, a_now + (1.0 - follow) * float(np.interp(a_now, BLEND_MARGIN_BP, margin)))

  @fail_safe(None)
  def plan(self, cruise_target: float, a_lead: float, a_model: float, a_lead_ahead: float, a_model_ahead: float,
           lead_distance: float, fcw: bool) -> tuple[float, float] | None:
    """
    cruise_target: from cruise_target()
    a_lead, a_model: the targets for right now
    a_lead_ahead, a_model_ahead: the same, read ahead by the lookahead
    lead_distance: to the closest lead, inf without one
    Returns the accel target and the smoothed cruise target, None when smoothing isn't possible.
    """
    if not np.isfinite([cruise_target, a_lead, a_model, a_lead_ahead, a_model_ahead, self.v_ego]).all():
      return None

    self.output_last = float(self.outputs[0])
    a_now = min(a_lead, a_model)
    a_cruise = self.cruise_accel(cruise_target, a_now)
    # it continues from its own targets like without smoothing, and has to know about the smoothed cruise
    self.a_plan = float(np.clip(min(a_cruise, a_lead, a_model), ACCEL_MIN, ACCEL_MAX))

    output = self.blend.update(float(np.clip(min(a_cruise, a_lead_ahead, a_model_ahead), ACCEL_MIN, ACCEL_MAX)))

    crawling = smoothstep(self.v_ego, *CRAWL_SPEED)
    if output < a_now < a_cruise:
      output += crawling * (a_now - output)

    has_lead = np.isfinite(lead_distance)
    if has_lead:
      creeping = crawling * smoothstep(lead_distance, *CREEP_DISTANCE) * smoothstep(a_lead, *CREEP_ACCEL) * smoothstep(self.v_ego, *CREEP_SPEED)
      output += creeping * min(-CREEP_DECEL - output, 0.0)

    # last, so that nothing above can get past it
    output = self._limit(output, a_model, BLEND_MARGIN_V, crawling)
    output = self._limit(output, a_lead, LEAD_MARGIN_V if has_lead else BLEND_MARGIN_V, 1.0 if fcw else crawling)
    output = float(np.clip(output, ACCEL_MIN, ACCEL_MAX))
    if not np.isfinite(output):
      return None

    self.outputs = np.roll(self.outputs, 1)
    self.outputs[0] = output
    return output, a_cruise
