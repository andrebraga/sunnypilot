"""
Copyright (c) 2021-, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np

from openpilot.common.realtime import DT_MDL

# Gap in Stopped Traffic: a stop that ends further back than the gap, behind a car that stands, closes up at a walk.
# Stops end short because the approach is planned for the stock distance. Standing still nothing else closes the gap:
# the lead plan keeps the stock distance until this has checked the scene
SETTLE_TIME = 1.5  # s, standing still before it closes up, the queue has settled
MIN_EXCESS = 1.0  # m, further back than the gap than this
MAX_EXCESS = 2.5  # m, but no more. further back than that the model holds for a reason (a crossing, a junction box)
SPEED = 0.45  # m/s, at most (1.6 km/h)
ACCEL = 0.3  # m/s^2, to get rolling
DECEL = 0.4  # m/s^2, it plans to stop at the gap with this
BRAKE = 1.5  # m/s^2, at most, to hold to that plan (downhill)
RESPONSE = 0.4  # s, of the car, rolled before braking takes hold
SPEED_TAU = 0.5  # s
ABORT_DECEL = -1.0  # m/s^2, stopping when the scene changes
FIRM_BRAKE = -0.3  # m/s^2, anything braking harder than this wins over it
LEAD_STILL = 0.3  # m/s, the car ahead stands
LEAD_JUMP = 0.5  # m, the car ahead coming closer than it can by us rolling: something else is there
MODEL_PROB = 0.5  # the camera sees the car the radar does
MAX_TIME = 8.0  # s


class CreepUp:
  def __init__(self, dt: float = DT_MDL):
    self.dt = dt
    self.standing = 0.0
    self.active = False
    self.stopping = False
    self.done = False
    self.time = 0.0
    self.travelled = 0.0
    self.start_gap = 0.0

  def reset(self) -> None:
    self.__init__(self.dt)

  @staticmethod
  def lead_ok(lead) -> bool:
    return bool(lead.present and lead.radar and lead.modelProb > MODEL_PROB and abs(lead.vLead) < LEAD_STILL)

  def abort(self, v_ego: float, a_target: float, lead) -> tuple[float, bool] | None:
    self.active = False
    if lead.present and lead.vLead > LEAD_STILL and not self.stopping:  # the queue moves: follow it like any other time
      self.done = True
      return None
    if v_ego < 0.05:
      self.stopping, self.done = False, True
      return None
    self.stopping = True
    return min(a_target, ABORT_DECEL), v_ego < 0.3

  def update(self, v_ego: float, standstill: bool, a_target: float, lead, gap: float, enabled: bool, fcw: bool,
             pedal: bool) -> tuple[float, bool] | None:
    """
    a_target: what everything else asks for
    lead: the closest lead from radard
    gap: where stops behind a lead should end, nan when Gap in Stopped Traffic is stock
    Returns the accel target and should_stop while closing up, None otherwise
    """
    if not enabled or fcw or pedal or not np.isfinite(v_ego) or not np.isfinite(a_target):
      self.reset()
      return None
    if self.stopping:
      return self.abort(v_ego, a_target, lead)
    if not np.isfinite(gap):
      self.reset()
      return None

    if v_ego > 3.0 * SPEED and not self.active:  # moving like traffic again: the next stop can close up
      self.standing = 0.0
      self.done = False
      return None
    self.standing = self.standing + self.dt if standstill else 0.0

    if not self.active:
      if self.done or self.standing < SETTLE_TIME or not self.lead_ok(lead) or a_target < FIRM_BRAKE:
        return None
      if not MIN_EXCESS <= lead.dRel - gap <= MAX_EXCESS:
        return None
      self.active, self.time, self.travelled, self.start_gap = True, 0.0, 0.0, float(lead.dRel)

    self.time += self.dt
    self.travelled += max(v_ego, 0.0) * self.dt
    something_closer = lead.present and lead.dRel < self.start_gap - self.travelled - LEAD_JUMP
    if not self.lead_ok(lead) or something_closer or a_target < FIRM_BRAKE or self.time > MAX_TIME or v_ego > 2.0 * SPEED:
      return self.abort(v_ego, a_target, lead)

    excess = float(lead.dRel - gap) - v_ego * RESPONSE
    if excess <= 0.0 and v_ego < 0.05:
      self.active, self.done = False, True
      return None
    v_target = min(SPEED, float(np.sqrt(2.0 * DECEL * max(excess, 0.0))))
    accel = float(np.clip((v_target - v_ego) / SPEED_TAU, -BRAKE, ACCEL))
    return accel, bool(v_target < 0.05 and v_ego < 0.3)
