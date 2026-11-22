"""
Copyright (c) 2021-, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np

from openpilot.cereal import custom, log
from opendbc.car.interfaces import ACCEL_MAX, ACCEL_MIN
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
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
CRUISE_DECEL = 0.5  # m/s^2, the driver lowered the set speed
VISION_DECEL = 1.0  # m/s^2, largest decel the vision controller asks for
APPROACH_DECEL = 1.2  # m/s^2, slowing for something at a known distance
APPROACH_DECEL_LATE = 2.0  # m/s^2, same but it was seen late
ARRIVAL_TAUS = 1.5  # cruise time constants until a slowdown counts as arrived

# Smoothing fades out as the unsmoothed lead and model targets ask for real braking, m/s^2
URGENT_LEAD = (-0.3, -0.8)
URGENT_LEAD_GAP = (0.05, 0.4)  # how much less the car would brake than asked for
CRAWL_SPEED = (4.0, 2.0)  # m/s, stick to the lead and model targets to pull away and stop as quickly as without smoothing
CREEP_DISTANCE = (4.5, 3.5)  # m, closer than where a stop should end. don't keep rolling on a slope with the brakes barely on
CREEP_ACCEL = (0.1, -0.05)  # m/s^2, when the lead plan doesn't want to go
CREEP_SPEED = (0.02, 0.2)  # m/s
CREEP_DECEL = 0.4  # m/s^2
URGENT_MODEL = (-1.0, -2.0)
URGENT_MODEL_GAP = (0.8, 1.6)
MODEL_LOOKAHEAD_DECEL = 0.5  # m/s^2, how much earlier braking to take from the model's plan

# The car's own response to hills, wind and so on
DISTURBANCE_TAU = 1.5  # s
DISTURBANCE_MAX = 1.0  # m/s^2

# Let go of the brakes just before stopping
FINAL_STOP_SPEED = 2.5  # m/s
FINAL_STOP_DECEL = (0.3, 1.6)  # m/s^2, at a stop and at FINAL_STOP_SPEED
FINAL_STOP_LEAD_DISTANCE = (12.0, 8.0)  # m, not behind a lead, its plan already stops softly
FINAL_STOP_ACCEL = (-1.5, -2.5)  # m/s^2, and when asked for harder braking


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
  """
  def __init__(self, CP, dt: float = DT_MDL):
    self.params = Params()
    self.dt = dt
    self.openpilot_longitudinal = CP.openpilotLongitudinalControl
    self.actuator_delay = CP.longitudinalActuatorDelay
    self.frame = -1
    self.enabled = False
    self.active = False

    self.ease_time = get_ease_time()
    self.cruise = Window(EASE_TIME_MAX, dt)
    self.cruise.set_kernel(ease_kernel(self.ease_time, EASE_ONSET, dt))
    self.blend = Window(BLEND_TIME, dt)
    self.blend.set_kernel(ease_kernel(BLEND_TIME, BLEND_ONSET, dt))
    self.n_delay = int(round(self.actuator_delay / dt))
    self.outputs = np.zeros(max(self.n_delay, 1))  # newest first
    self.a_plan = 0.0
    self.output_last = 0.0
    self.v_ego = 0.0
    self.disturbance = 0.0

  def update_params(self) -> None:
    self.frame += 1
    if self.frame % int(PARAMS_UPDATE_PERIOD / self.dt) == 0:
      self.enabled = self.openpilot_longitudinal and self.params.get_bool("LimoMode")

  def reset(self, accel: float) -> None:
    self.cruise.reset(accel)
    self.blend.reset(accel)
    self.outputs[:] = accel
    self.a_plan = accel
    self.disturbance = 0.0

  def start_frame(self, v_ego: float, a_ego: float, personality, active: bool) -> None:
    self.v_ego = v_ego
    self.active = active
    ease_time = get_ease_time(personality)
    if ease_time != self.ease_time:
      self.ease_time = float(np.clip(ease_time, self.ease_time - EASE_TIME_RATE * self.dt, self.ease_time + EASE_TIME_RATE * self.dt))
      self.cruise.set_kernel(ease_kernel(self.ease_time, EASE_ONSET, self.dt))
    if active and v_ego > FINAL_STOP_SPEED:
      disturbance = float(np.clip(a_ego - self.outputs[-1], -DISTURBANCE_MAX, DISTURBANCE_MAX))
      self.disturbance += (disturbance - self.disturbance) * self.dt / DISTURBANCE_TAU

  @property
  def lookahead(self) -> float:
    """How much further ahead to read the lead and model plans"""
    return self.blend.delay + LEAD_LOOKAHEAD

  def _pending(self, window: Window) -> float:
    return float((window.history + self.disturbance) @ window.pending) * self.dt

  @property
  def _pending_actuators(self) -> float:
    return float((self.outputs[:self.n_delay] + self.disturbance).sum()) * self.dt

  @property
  def committed_speed(self) -> float:
    """Speed after everything that's already on its way has happened"""
    return self.v_ego + self._pending(self.cruise) + self._pending(self.blend) + self._pending_actuators

  def approach_distance(self, v_target: float, decel: float = APPROACH_DECEL) -> float:
    """Distance to slow down to v_target at decel, given everything the car is already committed to"""
    dv = max(self.committed_speed - v_target, 0.0)

    # constant decel followed by an exponential tail
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

  def approach_decel(self, v_target: float, distance: float) -> float:
    """Lowest decel to be at v_target in distance"""
    if distance <= 0.0 or self.approach_distance(v_target, APPROACH_DECEL) <= distance:
      return APPROACH_DECEL
    if self.approach_distance(v_target, APPROACH_DECEL_LATE) >= distance:
      return APPROACH_DECEL_LATE

    low, high = APPROACH_DECEL, APPROACH_DECEL_LATE
    for _ in range(8):
      mid = (low + high) / 2.0
      if self.approach_distance(v_target, mid) > distance:
        low = mid
      else:
        high = mid
    return high

  def cruise_target(self, v_cruise: float, source, distance: float) -> float:
    """Unsmoothed cruise target"""
    if source in (LongitudinalPlanSource.sccMap, LongitudinalPlanSource.speedLimitAssist):
      decel = self.approach_decel(v_cruise, distance)
    elif source == LongitudinalPlanSource.sccVision:
      decel = VISION_DECEL
    else:
      decel = CRUISE_DECEL
    accel = float(np.clip((v_cruise - self.committed_speed) / CRUISE_TAU, -decel, ACCEL_MAX))
    return float(np.clip(accel - self.disturbance, ACCEL_MIN, ACCEL_MAX))

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
    target = float(np.clip(target, last - TARGET_STEP, last + TARGET_STEP)) if np.isfinite(target) else last
    return self.cruise.update(target)

  def update_plan(self, a_cruise: float, a_lead: float, a_model: float) -> float:
    """
    The acceleration the lead MPC continues from. It expects its own targets back, like without smoothing, and has to
    know about the smoothed cruise.
    """
    self.a_plan = float(np.clip(min(a_cruise, a_lead, a_model), ACCEL_MIN, ACCEL_MAX))
    return self.a_plan

  @staticmethod
  def _urgent(output: float, a_now: float, accel: tuple[float, float], gap: tuple[float, float], urgency: float = 0.0) -> float:
    if a_now >= output:
      return output
    urgency = max(urgency, smoothstep(a_now, *accel), smoothstep(output - a_now, *gap))
    return output + urgency * (a_now - output)

  @staticmethod
  def _soft_stop(accel: float, v_next: float, strength: float) -> float:
    a_min = -float(np.interp(v_next, [0.0, FINAL_STOP_SPEED], FINAL_STOP_DECEL))
    if v_next >= FINAL_STOP_SPEED or accel >= a_min:
      return accel
    return accel + strength * (a_min - accel)

  def update(self, a_target: float, a_cruise: float, a_lead: float | None, a_model: float, lead_distance: float, fcw: bool) -> float:
    """
    a_target: lowest of all targets, with cruise smoothed and the plans read ahead by the lookahead
    a_cruise: smoothed cruise target
    a_lead, a_model: the targets for right now. a_lead is None without a lead
    """
    self.output_last = float(self.outputs[0])
    a_target = float(np.clip(a_target, ACCEL_MIN, ACCEL_MAX)) if np.isfinite(a_target) else self.output_last
    a_now = min(a_model, ACCEL_MAX if a_lead is None else a_lead)

    soft_stop = 0.0 if fcw else (1.0 - smoothstep(a_now, *FINAL_STOP_ACCEL)) * (1.0 - smoothstep(lead_distance, *FINAL_STOP_LEAD_DISTANCE))
    v_next = self.v_ego + self._pending_actuators
    output = self.blend.update(self._soft_stop(a_target, v_next + self._pending(self.blend), soft_stop))

    if a_model < 0.0:
      output = self._urgent(output, a_model, URGENT_MODEL, URGENT_MODEL_GAP)
    crawling = smoothstep(self.v_ego, *CRAWL_SPEED)
    if a_lead is not None:
      output = self._urgent(output, a_lead, URGENT_LEAD, URGENT_LEAD_GAP, 1.0 if fcw else crawling)
    if 0.0 < a_now < a_cruise and output < a_now:
      output += crawling * (a_now - output)
    output = self._soft_stop(output, v_next, soft_stop)
    if a_lead is not None:
      creeping = crawling * smoothstep(lead_distance, *CREEP_DISTANCE) * smoothstep(a_lead, *CREEP_ACCEL) * smoothstep(self.v_ego, *CREEP_SPEED)
      output += creeping * min(-CREEP_DECEL - output, 0.0)

    output = float(np.clip(output, ACCEL_MIN, ACCEL_MAX))
    self.outputs = np.roll(self.outputs, 1)
    self.outputs[0] = output
    return output
