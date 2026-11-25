from types import SimpleNamespace

import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, STOP_DISTANCE, CLOSE_STOP_SPEED, T_IDXS
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import STOP_GAP_MARGIN, STOP_GAP_MIN, stop_distance_from_param


def lead(d, v=0.0):
  return SimpleNamespace(present=True, dRel=d, vLead=v, vLeadK=v, aLeadK=0.0, aLeadTau=1.5, modelProb=1.0, radar=True)


def no_lead():
  return SimpleNamespace(present=False)


def stop_behind(stop_distance, v0=8.0, gap=40.0, steps=600):
  """Plans from v0 towards a stopped car and follows the plan, returns the final gap"""
  mpc = LongitudinalMpc()
  mpc.stop_distance = stop_distance
  v, a, x = v0, 0.0, 0.0
  for _ in range(steps):
    mpc.set_cur_state(v, a)
    mpc.update(SimpleNamespace(leadOne=lead(gap - x), leadTwo=no_lead()))
    v_next = max(float(np.interp(0.05, T_IDXS, mpc.v_solution)), 0.0)
    a = float(np.interp(0.05, T_IDXS, mpc.a_solution))
    x += (v + v_next) / 2 * 0.05
    v = v_next
  return gap - x


class TestStopDistance(OpenpilotTestCase):
  def test_param(self):
    for bad in (None, 0, "0", "x", -5, float("nan")):
      assert stop_distance_from_param(bad) == STOP_DISTANCE
    assert stop_distance_from_param(30) == 3.0 + STOP_GAP_MARGIN
    assert stop_distance_from_param("25") == 2.5 + STOP_GAP_MARGIN
    assert stop_distance_from_param(5) == STOP_GAP_MIN + STOP_GAP_MARGIN
    assert stop_distance_from_param(200) == STOP_DISTANCE

  def test_shift_only_at_a_crawl(self):
    mpc = LongitudinalMpc()
    assert mpc.stop_distance_shift(10.0) == 0.0
    mpc.stop_distance = 4.0
    for v, expected in ((0.0, 2.0), (CLOSE_STOP_SPEED[0], 2.0), (CLOSE_STOP_SPEED[1], 0.0), (30.0, 0.0)):
      mpc.reset()
      mpc.set_cur_state(v, 0.0)
      for _ in range(200):
        shift = mpc.stop_distance_shift(100.0)
      assert np.isclose(shift, expected)

  def test_not_when_it_takes_braking(self):
    # 1.5 m/s with 1 m to spare before the closer gap: stock margin
    mpc = LongitudinalMpc()
    mpc.stop_distance = 4.0
    mpc.set_cur_state(1.5, 0.0)
    for _ in range(200):
      assert mpc.stop_distance_shift(5.0) == 0.0
    # already past it
    assert mpc.stop_distance_shift(3.0) == 0.0
    for bad in (np.nan, -np.inf):
      assert mpc.stop_distance_shift(bad) == 0.0

  def test_not_standing_still(self):
    # standing behind a car only the creep up may close the gap: the lead plan keeps the stock distance
    mpc = LongitudinalMpc()
    mpc.stop_distance = 3.0
    mpc.set_cur_state(0.0, 0.0)
    for _ in range(50):
      mpc.stop_distance_shift(10.0)
    assert mpc.stop_shift > 0.0
    mpc.close_in = False
    assert mpc.stop_distance_shift(10.0) == 0.0

  def test_closes_in_slowly_opens_at_once(self):
    mpc = LongitudinalMpc()
    mpc.stop_distance = 4.0
    mpc.set_cur_state(0.5, 0.0)
    shifts = [mpc.stop_distance_shift(10.0) for _ in range(5)]
    assert 0.0 < shifts[0] < shifts[-1] < 1.0
    assert mpc.stop_distance_shift(4.1) == 0.0

  def test_stops_closer(self):
    stock = stop_behind(STOP_DISTANCE)
    closer = stop_behind(4.0)
    # a gentle stop ends at about the closer distance
    assert stock - closer > 0.7 and abs(closer - 4.0) < 0.4, (stock, closer)

  def test_same_at_speed(self):
    # following at speed is not changed
    mpc_a, mpc_b = LongitudinalMpc(), LongitudinalMpc()
    mpc_b.stop_distance = 3.5
    for mpc in (mpc_a, mpc_b):
      for _ in range(20):
        mpc.set_cur_state(25.0, 0.0)
        mpc.update(SimpleNamespace(leadOne=lead(50.0, 25.0), leadTwo=no_lead()))
    assert np.allclose(mpc_a.a_solution, mpc_b.a_solution)
