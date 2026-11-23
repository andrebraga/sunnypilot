"""
Copyright (c) 2021-, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import itertools

import numpy as np

from openpilot.cereal import custom, log
from opendbc.car.interfaces import ACCEL_MAX, ACCEL_MIN
from openpilot.common.parameterized import parameterized
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.selfdrive.controls.lib.limo_curve import DEFAULTS, PARAM, PERSONALITIES, RANGES, LimoCurve, curves_from_param, \
  curves_to_param, get_curve
from openpilot.sunnypilot.selfdrive.controls.lib.limo_mode import LimoMode, preview
from openpilot.sunnypilot.selfdrive.controls.lib.tests.test_limo_mode import Car, drive, make_limo

LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource
Personality = log.LongitudinalPersonality

EXTREMES = [LimoCurve(*v) for v in itertools.product(*((r.min, r.max) for r in RANGES.values()))]
CURVES = [(c,) for c in EXTREMES + list(DEFAULTS.values())]


def make_curved(curve: LimoCurve) -> LimoMode:
  limo = make_limo()
  limo.curves = dict.fromkeys(PERSONALITIES, curve)
  limo.curve = curve
  limo.set_curve_kernels()
  return limo


class TestLimoCurve(OpenpilotTestCase):
  @parameterized.expand([(None,), ("",), ([],), (3,), ({},), ({"aggressive": None},), ({"standard": {"strength": "a lot"}},),
                         ({"relaxed": {"build": float("nan"), "ease": None, "strength": True}},), ({"relaxed": {"build": float("inf")}},)])
  def test_garbage_is_the_default(self, value):
    assert curves_from_param(value) == DEFAULTS

  def test_out_of_range_is_clipped(self):
    curve = curves_from_param({"standard": {"strength": 50.0, "build": -3, "ease": 1e9}})["standard"]
    assert (curve.strength, curve.build, curve.ease) == (ACCEL_MAX, RANGES["build"].min, RANGES["ease"].max)
    assert curves_from_param({"standard": {"strength": -5}})["standard"].strength == RANGES["strength"].min

  def test_param(self):
    curves = dict(DEFAULTS, relaxed=LimoCurve(0.85, 1.25, 3.1))
    Params().put(PARAM, curves_to_param(curves), block=True)
    assert curves_from_param(Params().get(PARAM)) == curves
    limo = make_limo()
    limo.update_params()
    assert get_curve(limo.curves, Personality.relaxed) == curves["relaxed"]
    Params().remove(PARAM)
    assert curves_from_param(Params().get(PARAM)) == DEFAULTS

  def test_personalities(self):
    for i, name in enumerate(PERSONALITIES):
      assert get_curve(DEFAULTS, i) == DEFAULTS[name]
      assert log.LongitudinalPersonality.schema.enumerants[name] == i

  def test_sporty_changes_nothing_about_the_lead_and_model(self):
    limo = make_curved(DEFAULTS["aggressive"])
    for a in np.random.default_rng(0).uniform(ACCEL_MIN, ACCEL_MAX, 300):
      assert limo.limit_rise(a) == a

  @parameterized.expand(CURVES)
  def test_only_takes_away_from_accelerating(self, curve):
    limo = make_curved(curve)
    rng = np.random.default_rng(0)
    a = 0.0
    for _ in range(600):
      a = float(np.clip(a + rng.uniform(-1.0, 1.0), ACCEL_MIN, ACCEL_MAX)) if rng.random() < 0.9 else float(rng.uniform(ACCEL_MIN, ACCEL_MAX))
      out = limo.limit_rise(a)
      assert out == a if a <= 0.0 else 0.0 <= out <= min(a, curve.strength) + 1e-9

  @parameterized.expand(CURVES)
  def test_never_brakes_less_or_accelerates_more_than_asked(self, curve):
    rng = np.random.default_rng(1)
    for _ in range(6):
      limo = make_curved(curve)
      v_ego, a_lead, a_model, has_lead = rng.uniform(0.0, 35.0), 1.0, 1.0, True
      for _ in range(150):
        if rng.random() < 0.1:
          a_lead, a_model = rng.uniform(-4.0, 3.0, 2)
          has_lead = rng.random() < 0.7
        v_ego = max(v_ego + rng.uniform(-0.3, 0.3), 0.0)
        limo.start_frame(v_ego, rng.uniform(-3.0, 2.0), Personality.standard, True)
        target = limo.cruise_target(rng.uniform(0.0, 40.0), LongitudinalPlanSource.cruise)
        ahead = rng.uniform(-4.0, 3.0, 2)
        accel, _ = limo.plan(target, a_lead, a_model, *ahead, rng.uniform(1.0, 60.0) if has_lead else np.inf, False)
        asked = max(min(a_lead, a_model), ACCEL_MIN)
        assert ACCEL_MIN <= accel <= curve.strength + 1e-6
        assert accel <= asked + float(np.interp(asked, [-0.5, 0.0, 0.5], [0.0, 0.2, 0.5])) + 1e-6

  @parameterized.expand(CURVES)
  def test_reaches_and_holds_the_set_speed(self, curve):
    limo, car = make_curved(curve), Car(10.0)
    v = [car.v]
    for _ in range(int(60.0 / DT_MDL)):
      drive(limo, car, 1, 20.0)
      v.append(car.v)
    assert abs(car.v - 20.0) < 0.1 and max(v) < 20.4

  @parameterized.expand(CURVES)
  def test_arrives_at_target_speed(self, curve):
    limo, car = make_curved(curve), Car(28.0)
    drive(limo, car, 800, 28.0)
    start = car.x + 1500.0
    approaching = False
    while car.x < start:
      approaching = approaching or start - car.x < limo.approach_distance(18.0)
      drive(limo, car, 1, 18.0 if approaching else 28.0, LongitudinalPlanSource.sccMap if approaching else LongitudinalPlanSource.cruise)
    assert abs(car.v - 18.0) < 0.5

  @parameterized.expand(CURVES)
  def test_model_launch_is_as_gentle_as_the_curve(self, curve):
    limo, car = make_curved(curve), Car(0.0)
    for _ in range(40):
      drive(limo, car, 1, 25.0, a_model=-0.5)
      car.v = 0.0
    out = drive(limo, car, 200, 25.0, a_model=ACCEL_MAX)
    assert out.max() <= curve.strength + 1e-6
    dips = np.diff(out[2:60])
    assert dips[dips < 0.0].sum() > -0.03
    if curve.build >= 1.0:
      assert np.diff(out[2:]).max() / DT_MDL < 1.6 * curve.strength
    assert out[-1] > 0.95 * curve.strength

  def test_changing_the_curve_does_not_step(self):
    for a, b in itertools.permutations(EXTREMES, 2):
      limo, car = make_curved(a), Car(5.0)
      drive(limo, car, 60, 40.0)
      limo.curves = dict.fromkeys(PERSONALITIES, b)
      out = drive(limo, car, 160, 40.0)
      assert np.abs(np.diff(out)).max() / DT_MDL < 4.0
      assert limo.curve == b

  def test_braking_is_not_delayed_after_accelerating(self):
    for curve in EXTREMES:
      limo, car = make_curved(curve), Car(15.0)
      drive(limo, car, 200, 30.0)
      assert drive(limo, car, 1, 30.0, a_lead=-2.5)[0] <= -2.5 + 1e-6

  @parameterized.expand(CURVES)
  def test_lets_go_when_the_lead_does(self, curve):
    limo, car = make_curved(curve), Car(15.0)
    drive(limo, car, 200, 30.0, a_lead=1.0)
    drive(limo, car, 40, 30.0, a_lead=-1.5)
    out = drive(limo, car, 60, 30.0, a_lead=1.0)
    # the blend window rounds it off, nothing holds on to the braking
    assert out[int(0.2 / DT_MDL)] > -0.1
    dips = np.diff(out[:20])
    assert dips[dips < 0.0].sum() > -0.03

  def test_stay_out(self):
    limo = make_curved(LimoCurve(0.8, 1.0, 3.0))
    out = [limo.stay_out(2.0) for _ in range(100)]
    assert (np.diff(out) >= -1e-9).all() and abs(out[-1] - 0.8) < 1e-6
    assert limo.stay_out(-1.3) == -1.3
    assert limo.stay_out(float("nan")) is None
    assert np.isfinite(limo.stay_out(1.0))

  @parameterized.expand(CURVES)
  def test_preview(self, curve):
    speed, accel = preview(curve, 13.9, 30.0)
    assert len(speed) == len(accel) == 601 and np.isfinite(speed).all()
    assert accel.max() <= curve.strength + 1e-6 and speed.max() < 14.2
