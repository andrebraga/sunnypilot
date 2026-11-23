"""
Copyright (c) 2021-, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from collections import deque
from types import SimpleNamespace

import numpy as np

from openpilot.cereal import custom, log
from opendbc.car.interfaces import ACCEL_MAX, ACCEL_MIN
from openpilot.common.parameterized import parameterized
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.selfdrive.controls.lib.limo_curve import DEFAULTS, ease_kernel
from openpilot.sunnypilot.selfdrive.controls.lib.limo_mode import LimoMode, BLEND_ONSET, BLEND_TIME, CRUISE_DECEL, CRUISE_HEADROOM, \
  DISTURBANCE_MAX, VISION_TIME, smoothstep

LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource
Personality = log.LongitudinalPersonality

DELAY = 0.5


def make_limo(long=True, delay=DELAY):
  limo = LimoMode(SimpleNamespace(openpilotLongitudinalControl=long, longitudinalActuatorDelay=delay))
  limo.enabled = True
  return limo


class Car:
  """A car that does what it's told, late"""
  def __init__(self, v, grade=0.0):
    self.v, self.a, self.x, self.grade = v, grade, 0.0, grade
    self.commands = deque([0.0] * int(round(DELAY / DT_MDL)))

  def step(self, accel):
    self.commands.append(accel)
    self.a = self.commands.popleft() + self.grade
    self.x += self.v * DT_MDL + 0.5 * self.a * DT_MDL ** 2
    self.v += self.a * DT_MDL


def drive(limo, car, frames, v_cruise, source=LongitudinalPlanSource.cruise, a_lead=None, a_model=ACCEL_MAX,
          personality=Personality.standard):
  out = []
  for _ in range(frames):
    limo.start_frame(car.v, car.a, personality, True)
    a_lead_now = ACCEL_MAX if a_lead is None else a_lead
    cruise_target = limo.cruise_target(v_cruise, source)
    accel, _ = limo.plan(cruise_target, a_lead_now, a_model, a_lead_now, a_model, 50.0 if a_lead is not None else np.inf, False)
    car.step(accel)
    out.append(accel)
  return np.array(out)


class TestLimoMode(OpenpilotTestCase):
  @parameterized.expand([(c.ease_time, c.build) for c in DEFAULTS.values()] + [(BLEND_TIME, BLEND_ONSET)])
  def test_kernel(self, length, onset):
    kernel = ease_kernel(length, onset, DT_MDL)
    assert len(kernel) == round(length / DT_MDL)
    assert abs(kernel.sum() - 1.0) < 1e-9 and (kernel > 0.0).all()
    assert abs((np.argmax(kernel) + 0.5) * DT_MDL - onset) <= DT_MDL
    # front loaded, and it fades in and out
    assert kernel @ np.arange(len(kernel)) * DT_MDL < 0.45 * length
    assert kernel[-1] < 0.2 * kernel.max()

  def test_window_keeps_the_sum(self):
    limo = make_limo()
    rng = np.random.default_rng(0)
    inputs = np.append(rng.uniform(-2, 2, 100), np.zeros(100))
    out = []
    for i, value in enumerate(inputs):
      out.append(limo.cruise.update(value))
      # what went in is either out already or still pending
      assert abs(inputs[:i + 1].sum() - sum(out) - limo.cruise.history @ limo.cruise.pending) < 1e-9
    moment = (np.array(out) - inputs) @ np.arange(200) * DT_MDL
    assert abs(moment - limo.cruise.delay * inputs.sum()) < 1e-6

  @parameterized.expand([(Personality.aggressive, 0.0), (Personality.standard, 0.0), (Personality.aggressive, 15.0), (Personality.standard, 15.0)])
  def test_does_not_start_later(self, personality, v0):
    # cruise without smoothing is limited to this jerk
    jerk = np.interp(v0, [0., 10.0, 25., 40.], [1.6, 1.2, 0.8, 0.6])
    limo, car = make_limo(), Car(v0)
    for _ in range(200):
      limo.start_frame(v0, 0.0, personality, True)
    out = drive(limo, car, 40, v0 + 20.0, personality=personality)
    reference = np.minimum(jerk * DT_MDL * np.arange(1, 41), ACCEL_MAX)
    assert out[0] > 0.0
    assert (np.cumsum(out - reference) * DT_MDL).min() > -0.06

  def test_pulls_away_with_the_lead_and_model(self):
    for a_lead, a_model in ((True, False), (False, True)):
      limo, car = make_limo(), Car(0.0)
      drive(limo, car, 60, 20.0, a_lead=0.0 if a_lead else None, a_model=0.0)
      car.v = car.a = 0.0
      for target in (0.3, 0.6, 0.9, 0.9):
        out = drive(limo, car, 1, 20.0, a_lead=target if a_lead else None, a_model=target if a_model else ACCEL_MAX)
        assert abs(out[0] - target) < 0.02

  def test_cruise_does_not_hold_back_boost(self):
    limo, car = make_limo(), Car(15.0)
    drive(limo, car, 100, 40.0, a_model=0.5)
    for boost in np.arange(0.0, 0.4, 0.2 * DT_MDL):
      a_model = 0.5 + boost
      out = drive(limo, car, 1, 40.0, a_model=a_model)
      assert limo.cruise.value > a_model
      assert abs(out[0] - a_model) < 0.04

  def test_cruise_follows_what_limits_it(self):
    limo, car = make_limo(), Car(15.0)
    drive(limo, car, 100, 40.0, a_model=-0.3)
    assert abs(limo.cruise.value - (-0.3 + CRUISE_HEADROOM)) < 1e-6

  def test_smoothstep(self):
    assert smoothstep(-0.2, -0.3, -0.8) == 0.0 and smoothstep(-0.9, -0.3, -0.8) == 1.0
    assert abs(smoothstep(-0.55, -0.3, -0.8) - 0.5) < 1e-9

  @parameterized.expand([(10.0, 25.0), (25.0, 33.0), (28.0, 30.0), (30.0, 25.0)])
  def test_reaches_set_speed_without_overshoot(self, v0, v_cruise):
    limo, car = make_limo(), Car(v0)
    v = []
    for _ in range(60 * 20):
      drive(limo, car, 1, v_cruise)
      v.append(car.v)
    assert abs(v[-1] - v_cruise) < 0.02
    overshoot = (max(v) - v_cruise) if v_cruise > v0 else (v_cruise - min(v))
    assert overshoot < 0.05

  @parameterized.expand([(p, v) for p in (Personality.aggressive, Personality.standard, Personality.relaxed) for v in (3.0, 15.0, 30.0)])
  def test_jerk_is_continuous(self, personality, dv):
    limo, car = make_limo(), Car(10.0)
    out = drive(limo, car, 800, 10.0, personality=personality)
    out = np.append(out, drive(limo, car, 800, 10.0 + dv, personality=personality))
    jerk = np.diff(out) / DT_MDL
    snap = np.diff(jerk) / DT_MDL
    # a step in jerk would be about max jerk / dt
    assert np.abs(snap).max() < 0.65 * np.abs(jerk).max() / DT_MDL

  def test_holds_full_accel_at_speed(self):
    limo, car = make_limo(), Car(25.0)
    out = drive(limo, car, 60, 45.0)
    assert out[-1] > 0.99 * ACCEL_MAX

  @parameterized.expand([(-0.3,), (-0.15,), (0.15,), (0.3,)])
  def test_no_steady_error_on_hills(self, grade):
    limo, car = make_limo(), Car(20.0, grade)
    drive(limo, car, 60 * 20, 20.0)
    assert abs(car.v - 20.0) < 0.02

  @parameterized.expand([(30.0, 20.0, 0.0), (25.0, 15.0, 0.0), (20.0, 12.0, 0.0), (30.0, 25.0, 0.0), (30.0, 20.0, -0.2), (30.0, 20.0, 0.2)])
  def test_arrives_at_target_speed(self, v0, v_target, grade):
    limo, car = make_limo(), Car(v0, grade)
    drive(limo, car, 600, v0)
    start = car.x + 1500.0
    approaching = False
    while car.x < start:
      approaching = approaching or start - car.x < limo.approach_distance(v_target)
      source = LongitudinalPlanSource.sccMap if approaching else LongitudinalPlanSource.cruise
      drive(limo, car, 1, v_target if approaching else v0, source)
    assert abs(car.v - v_target) < 0.35

  def test_set_speed_decel_is_gentle(self):
    limo, car = make_limo(), Car(30.0)
    assert drive(limo, car, 600, 20.0).min() > -0.55

  def test_model_lookahead_only_adds_braking(self):
    assert LimoMode.model_accel(0.5, 1.8, 0.8) == 0.5
    assert LimoMode.model_accel(0.5, 0.2, 0.9) == 0.5
    assert abs(LimoMode.model_accel(0.0, -0.1, -0.3) + 0.2) < 1e-9
    assert LimoMode.model_accel(0.0, 0.0, -3.0) == -0.5

  def test_personality_change_does_not_step(self):
    limo, car = make_limo(), Car(10.0)
    out = drive(limo, car, 15, 30.0, personality=Personality.relaxed)
    out = np.append(out, drive(limo, car, 60, 30.0, personality=Personality.aggressive))
    assert np.abs(np.diff(out)).max() / DT_MDL < 4.0

  def test_never_brakes_less_than_asked(self):
    rng = np.random.default_rng(0)
    for _ in range(20):
      limo = make_limo()
      v_ego = rng.uniform(0.0, 35.0)
      limo.reset(rng.uniform(-2.0, 2.0))
      a_lead, a_model, has_lead = 1.0, 1.0, True
      for _ in range(200):
        if rng.random() < 0.1:
          a_lead, a_model = rng.uniform(-4.0, 3.0, 2)
          has_lead = rng.random() < 0.7
        v_ego = max(v_ego + rng.uniform(-0.3, 0.3), 0.0)
        limo.start_frame(v_ego, rng.uniform(-3.0, 2.0), Personality.standard, True)
        target = limo.cruise_target(rng.uniform(0.0, 40.0), LongitudinalPlanSource.cruise)
        ahead = rng.uniform(-4.0, 3.0, 2)
        accel, _ = limo.plan(target, a_lead, a_model, *ahead, rng.uniform(1.0, 60.0) if has_lead else np.inf, rng.random() < 0.05)
        assert ACCEL_MIN <= accel <= ACCEL_MAX
        asked = max(min(a_lead, a_model), ACCEL_MIN)
        if asked <= -0.5:
          assert accel <= asked + 1e-6
        elif asked <= 0.0:
          assert accel <= asked + 0.2 + 1e-6
        else:
          assert accel <= asked + 0.5 + 1e-6

  def test_braking_is_immediate(self):
    for kwargs in ({"a_lead": -1.0}, {"a_lead": ACCEL_MIN}, {"a_model": -0.6}, {"a_model": -2.5}):
      limo, car = make_limo(), Car(25.0)
      drive(limo, car, 200, 30.0)
      assert drive(limo, car, 1, 30.0, **kwargs)[0] <= min(kwargs.values()) + 1e-6

  def test_never_brakes_harder_for_cruise_than_stock(self):
    for source in (LongitudinalPlanSource.cruise, LongitudinalPlanSource.sccMap, LongitudinalPlanSource.speedLimitAssist):
      for grade in (-0.4, 0.0, 0.4, 0.8):
        limo, car = make_limo(), Car(35.0, grade=grade)
        assert drive(limo, car, 400, 5.0, source).min() >= -CRUISE_DECEL - 1e-6

  def test_disturbance_is_bounded(self):
    limo = make_limo()
    for _ in range(400):
      limo.start_frame(20.0, -3.5, Personality.standard, True)
    assert abs(limo.disturbance) <= DISTURBANCE_MAX

  def test_stays_out_of_vision_turns(self):
    limo = make_limo()
    limo.start_frame(15.0, 0.0, Personality.standard, True)
    assert limo.cruise_target(10.0, LongitudinalPlanSource.sccVision) is None
    for _ in range(int(VISION_TIME / DT_MDL) - 1):
      assert limo.cruise_target(10.0, LongitudinalPlanSource.cruise) is None
    for _ in range(2):
      target = limo.cruise_target(10.0, LongitudinalPlanSource.cruise)
    assert target is not None

  @parameterized.expand([(i, bad) for i in range(5) for bad in (float("nan"), float("inf"))])
  def test_bad_input_falls_back(self, index, bad):
    limo, car = make_limo(), Car(20.0)
    drive(limo, car, 20, 20.0)
    args = [1.0, 1.0, 1.0, 1.0, 1.0]
    args[index] = bad
    limo.start_frame(20.0, 0.0, Personality.standard, True)
    assert limo.plan(*args, np.inf, False) is None
    # and it doesn't stick
    assert np.isfinite(drive(limo, car, 50, 20.0)).all() and not limo.failed

  def test_failure_is_latched(self):
    params = Params()
    params.put_bool("LimoMode", True, block=True)
    limo, car = make_limo(), Car(20.0)
    drive(limo, car, 20, 20.0)
    limo.cruise = None
    assert limo.plan(1.0, 1.0, 1.0, 1.0, 1.0, np.inf, False) is None
    assert limo.failed and not limo.enabled
    limo.frame = -1
    limo.update_params()
    assert not limo.enabled
    assert limo.approach_distance(10.0) == 0.0
    assert limo.cruise_target(10.0, LongitudinalPlanSource.cruise) is None

  @parameterized.expand([("cruise",), ("blend",), ("outputs",), ("params",)])
  def test_no_failure_gets_out(self, name):
    limo, car = make_limo(), Car(20.0)
    drive(limo, car, 20, 20.0)
    setattr(limo, name, None)
    for _ in range(3 * int(1.0 / DT_MDL)):
      limo.update_params()
      limo.start_frame(20.0, 0.0, log.LongitudinalPersonality.standard, True)
      assert limo.approach_distance(10.0) >= 0.0
      limo.cruise_target(20.0, LongitudinalPlanSource.cruise)
      limo.plan(0.0, 0.0, 0.0, 0.0, 0.0, 50.0, False)
      limo.reset(0.0)
    assert limo.failed and not limo.enabled

  @parameterized.expand([(np.nan,), (np.inf,)])
  def test_bad_state_does_not_become_a_target(self, bad):
    limo = make_limo()
    limo.outputs[:] = bad
    limo.disturbance = np.nan
    assert limo.cruise_target(20.0, LongitudinalPlanSource.cruise) is None
    limo.reset(0.0)
    assert limo.cruise_target(20.0, LongitudinalPlanSource.cruise) is not None

  @parameterized.expand([(3.0, -0.05, -0.4), (6.0, -0.05, -0.05), (3.0, 0.2, 0.2)])
  def test_stops_rolling_close_to_the_lead(self, distance, a_lead, expected):
    limo = make_limo()
    for _ in range(40):
      limo.start_frame(0.3, 0.0, Personality.standard, True)
      accel, _ = limo.plan(ACCEL_MAX, a_lead, ACCEL_MAX, a_lead, ACCEL_MAX, distance, False)
    assert abs(accel - expected) < 1e-3

  def test_starts_from_current_accel(self):
    limo = make_limo()
    limo.reset(-0.4)
    limo.start_frame(20.0, -0.4, Personality.standard, True)
    assert abs(limo.plan(-0.4, 2.0, 2.0, 2.0, 2.0, np.inf, False)[0] + 0.4) < 1e-6

  def test_param(self):
    params = Params()
    for long in (True, False):
      limo = LimoMode(SimpleNamespace(openpilotLongitudinalControl=long, longitudinalActuatorDelay=DELAY))
      assert not limo.enabled
      params.put_bool("LimoMode", True, block=True)
      limo.frame = -1
      limo.update_params()
      assert limo.enabled == long
      params.put_bool("LimoMode", False, block=True)
      limo.frame = -1
      limo.update_params()
      assert not limo.enabled
