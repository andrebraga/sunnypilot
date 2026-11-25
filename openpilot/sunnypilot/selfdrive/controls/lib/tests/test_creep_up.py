"""
Copyright (c) 2021-, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.selfdrive.controls.lib.creep_up import CreepUp, ABORT_DECEL, ACCEL, MAX_EXCESS, SETTLE_TIME, SPEED

GAP = 3.0


def lead(d=5.0, v=0.0, present=True, radar=True, prob=1.0):
  return SimpleNamespace(present=present, dRel=d, vLead=v, radar=radar, modelProb=prob)


class Car:
  def __init__(self, gap=5.0):
    self.v, self.x, self.gap0 = 0.0, 0.0, gap

  def step(self, accel):
    self.v = max(0.0, self.v + accel * DT_MDL)
    self.x += self.v * DT_MDL

  @property
  def gap(self):
    return self.gap0 - self.x


def run(c, car, seconds, a_target=0.0, lead_fn=None, enabled=True, fcw=False, pedal=False):
  outs = []
  for i in range(int(seconds / DT_MDL)):
    ld = lead_fn(car, i * DT_MDL) if lead_fn else lead(car.gap)
    out = c.update(car.v, car.v < 0.01, a_target, ld, GAP, enabled, fcw, pedal)
    car.step(out[0] if out is not None else (-2.0 if car.v > 0 else 0.0))
    outs.append(out)
  return outs


class TestCreepUp(OpenpilotTestCase):
  def test_closes_up_to_the_gap(self):
    c, car = CreepUp(), Car(5.0)
    run(c, car, 15.0)
    assert c.done and abs(car.gap - GAP) < 0.3
    assert car.v == 0.0

  def test_waits_for_the_queue_to_settle(self):
    c, car = CreepUp(), Car(5.0)
    outs = run(c, car, SETTLE_TIME - 0.2)
    assert all(o is None for o in outs) and car.x == 0.0

  def test_slow(self):
    c, car = CreepUp(), Car(GAP + MAX_EXCESS)
    vmax = 0.0
    for _ in range(int(20 / DT_MDL)):
      out = c.update(car.v, car.v < 0.01, 0.0, lead(car.gap), GAP, True, False, False)
      car.step(out[0] if out else 0.0)
      vmax = max(vmax, car.v)
      if out:
        assert out[0] <= ACCEL
    assert vmax <= SPEED + 0.05

  def test_not_when_close_enough(self):
    c, car = CreepUp(), Car(GAP + 0.8)
    run(c, car, 6.0)
    assert car.x == 0.0 and not c.active

  def test_not_from_far_back(self):
    # the model holds back that far for a reason: a crossing, a junction box
    c, car = CreepUp(), Car(GAP + MAX_EXCESS + 0.5)
    run(c, car, 6.0)
    assert car.x == 0.0 and not c.active

  def test_not_without_a_still_car_seen_by_both(self):
    for ld in (lead(present=False), lead(radar=False), lead(prob=0.2), lead(v=1.0)):
      c, car = CreepUp(), Car(5.0)
      run(c, car, 5.0, lead_fn=lambda car, t, ld=ld: SimpleNamespace(**{**vars(ld), 'dRel': car.gap}))
      assert car.x == 0.0

  def test_not_off_stock_or_firm_braking(self):
    for kw in ({"enabled": False}, {"fcw": True}, {"pedal": True}, {"a_target": -0.5}):
      c, car = CreepUp(), Car(5.0)
      run(c, car, 5.0, **kw)
      assert car.x == 0.0, kw
    c, car = CreepUp(), Car(5.0)
    assert all(c.update(0.0, True, 0.0, lead(5.0), np.nan, True, False, False) is None for _ in range(100))

  def test_stops_for_something_stepping_in(self):
    c, car = CreepUp(), Car(5.2)
    run(c, car, SETTLE_TIME + 1.5)
    assert c.active and car.v > 0.2
    x0 = car.x
    outs = run(c, car, 3.0, lead_fn=lambda car, t: lead(1.5 - (car.x - x0)))
    assert outs[0] is not None and outs[0][0] <= ABORT_DECEL
    assert car.v == 0.0 and car.x - x0 < 0.3

  def test_stops_when_the_radar_loses_it(self):
    c, car = CreepUp(), Car(5.2)
    run(c, car, SETTLE_TIME + 1.5)
    run(c, car, 3.0, lead_fn=lambda car, t: lead(present=False))
    assert car.v == 0.0 and c.done

  def test_follows_the_queue_when_it_moves(self):
    c, car = CreepUp(), Car(5.2)
    run(c, car, SETTLE_TIME + 1.5)
    assert c.active
    out = c.update(car.v, False, 0.0, lead(car.gap, v=1.0), GAP, True, False, False)
    assert out is None and c.done

  def test_once_per_stop(self):
    c, car = CreepUp(), Car(5.0)
    run(c, car, 15.0)
    assert c.done
    car.gap0 += 2.0  # the car ahead moved off a little and stopped, we didn't move
    run(c, car, 5.0)
    assert c.done and abs(car.gap - GAP - 2.0) < 0.3
