"""
Copyright (c) 2021-, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import json
import math
import platform
from types import SimpleNamespace
from unittest.mock import MagicMock

from openpilot.common.parameterized import parameterized
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_planner import A_CRUISE_MIN, get_cruise_accel
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP, LongitudinalPlanSource
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.map_controller import R, TARGET_ACCEL, SmartCruiseControlMap


def to_degrees(distance):
  return math.degrees(distance / R)


class TestCruiseAccelLimit(OpenpilotTestCase):
  def get_accel_min(self, source):
    planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
    planner.source = source
    return min(A_CRUISE_MIN, planner.a_target_min)

  @parameterized.expand([
    (LongitudinalPlanSource.cruise, A_CRUISE_MIN),
    (LongitudinalPlanSource.sccVision, A_CRUISE_MIN),
    (LongitudinalPlanSource.sccMap, TARGET_ACCEL),
  ])
  def test_source_limit(self, source, accel_min):
    assert self.get_accel_min(source) == accel_min

  def test_limit_is_used(self):
    for accel_min in (A_CRUISE_MIN, TARGET_ACCEL):
      a_cruise = 0.
      for _ in range(int(5. / DT_MDL)):
        a_cruise = get_cruise_accel(10., 30., a_cruise, DT_MDL, accel_min)
      assert a_cruise == accel_min

  @parameterized.expand([(30., 20.), (25., 15.), (20., 12.), (30., 26.)])
  def test_arrives_at_map_target(self, v_ego, v_target):
    params = Params()
    params.put_bool("SmartCruiseControlMap", True, block=True)
    mem_params = Params("/dev/shm/params") if platform.system() != "Darwin" else params
    target_distance = 600.
    points = [{"latitude": 0., "longitude": to_degrees(d), "velocity": v_target if d >= target_distance else 50.}
              for d in range(0, 800, 20)]
    mem_params.put("MapTargetVelocities", json.dumps(points), block=True)

    scc_map = SmartCruiseControlMap()
    gps = SimpleNamespace(latitude=0., longitude=0.)
    sm = MagicMock()
    sm.__getitem__.return_value = gps

    v_cruise = v_ego
    x = a_cruise = 0.
    while x < target_distance:
      gps.longitude = to_degrees(x)
      scc_map.update(sm, True, False, v_ego, a_cruise, v_cruise)
      source = LongitudinalPlanSource.sccMap if scc_map.output_v_target < v_cruise else LongitudinalPlanSource.cruise
      a_cruise = get_cruise_accel(min(v_cruise, scc_map.output_v_target), v_ego, a_cruise, DT_MDL, self.get_accel_min(source))
      v_ego += a_cruise * DT_MDL
      x += v_ego * DT_MDL
    mem_params.put("MapTargetVelocities", "{}", block=True)

    assert abs(v_ego - v_target) < 0.5
