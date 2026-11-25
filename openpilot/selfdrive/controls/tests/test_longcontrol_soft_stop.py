from types import SimpleNamespace

import numpy as np

from openpilot.common.realtime import DT_CTRL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState, SOFT_STOP_FLOOR, SOFT_STOP_SETTLE, SOFT_STOP_TIME

LIMITS = (-3.5, 2.0)


def car_params():
  CP = SimpleNamespace(stopAccel=-2.0, longitudinalTuning=SimpleNamespace(kiBP=[0.], kiV=[0.]))
  return CP, SimpleNamespace(enableGasInterceptor=False)


def state(v, standstill=None):
  return SimpleNamespace(vEgo=v, aEgo=0.0, brakePressed=False, standstill=v <= 0.104 if standstill is None else standstill,
                         cruiseState=SimpleNamespace(standstill=False))


def stop(soft, v0=0.29, a_target=-0.1, plant=True, frames=400, soft_at=lambda i: True):
  """Rolls into a stop from pid, returns (speed, output, state) per frame"""
  lc = LongControl(*car_params())
  v, a, out = v0, -0.2, []
  lc.update(True, state(v), -0.2, False, LIMITS, soft_stop=soft and soft_at(-1))
  lc.last_output_accel = -0.2
  for i in range(frames):
    u = lc.update(True, state(v), a_target, True, LIMITS, soft_stop=soft and soft_at(i))
    if plant:
      a += (u - a) * DT_CTRL / 0.25
      v = max(0.0, v + a * DT_CTRL) if not (v <= 0.0 and a < 0) else 0.0
    out.append((v, u, lc.long_control_state))
  return np.array(out, dtype=object)


class TestSoftStop(OpenpilotTestCase):
  def test_off_is_stock(self):
    a, b = stop(False), stop(False, soft_at=lambda i: False)
    assert np.array_equal(a[:, 1].astype(float), b[:, 1].astype(float))

  def test_lighter_at_the_halt(self):
    for soft in (False, True):
      o = stop(soft)
      v, u = o[:, 0].astype(float), o[:, 1].astype(float)
      halt = int(np.argmax(v <= 0.0))
      assert halt > 0
      if soft:
        assert u[halt - 1] > -0.4, u[halt - 1]
      else:
        assert u[halt - 1] < -0.6, u[halt - 1]

  def test_holds_after(self):
    # after it stands it squeezes for the hold like stock
    o = stop(True, frames=int((SOFT_STOP_SETTLE + 3.0) / DT_CTRL))
    assert -2.02 < float(o[-1, 1]) <= -2.0

  def test_never_less_than_plan(self):
    # (stock itself stops at stopAccel, -2.0)
    for a_target in (-0.3, -1.0, -1.9):
      o = stop(True, a_target=a_target, v0=0.29)
      u = o[:, 1].astype(float)
      ref = stop(False, a_target=a_target, v0=0.29)[:, 1].astype(float)
      # ramps towards the plan at the same rate as the stock squeeze does, so never later than stock gets there
      reached = np.argmax(u <= a_target + 1e-6)
      reached_stock = np.argmax(ref <= a_target + 1e-6)
      assert u.min() <= a_target + 1e-6 and reached <= reached_stock

  def test_time_cap(self):
    # a car that keeps rolling (no plant) gets the stock squeeze after SOFT_STOP_TIME
    o = stop(True, plant=False, frames=int((SOFT_STOP_TIME + 2.5) / DT_CTRL))
    u = o[:, 1].astype(float)
    assert u[int(SOFT_STOP_TIME / DT_CTRL) - 5] >= -0.5
    assert -2.02 < u[-1] <= -2.0

  def test_permission_lost_mid_stop(self):
    # a lead that comes close mid stop ends it for good: stock squeeze from there, even if it goes away again
    o = stop(True, plant=False, soft_at=lambda i: not 30 <= i < 40)
    u = o[:, 1].astype(float)
    assert np.all(np.diff(u[40:200]) <= 1e-9)
    assert u[199] < u[39] - 1.0

  def test_speeding_up_ends_it(self):
    lc = LongControl(*car_params())
    lc.update(True, state(0.25), -0.2, False, LIMITS, soft_stop=True)
    assert lc.soft_stopping
    for i in range(50):
      lc.update(True, state(0.25 + 0.002 * i), -0.1, True, LIMITS, soft_stop=True)
    assert not lc.soft_stopping

  def test_not_from_standstill(self):
    lc = LongControl(*car_params())
    lc.update(True, state(0.0), 0.0, True, LIMITS, soft_stop=True)
    assert lc.long_control_state == LongCtrlState.stopping and not lc.soft_stopping

  def test_floor(self):
    lc = LongControl(*car_params())
    lc.update(True, state(0.05, standstill=False), -0.2, False, LIMITS, soft_stop=True)
    for _ in range(100):
      u = lc.update(True, state(0.05, standstill=False), 0.0, True, LIMITS, soft_stop=True)
    assert abs(u + SOFT_STOP_FLOOR) < 1e-6, u
