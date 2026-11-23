"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import json
import math
import platform
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from openpilot.cereal import custom
from openpilot.common.parameterized import parameterized
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.map_controller import R, SmartCruiseControlMap
from openpilot.common.test import OpenpilotTestCase

MapState = VisionState = custom.LongitudinalPlanSP.SmartCruiseControl.MapState


class TestSmartCruiseControlMap(OpenpilotTestCase):

  def setup_method(self):
    self.params = Params()
    self.mem_params = Params("/dev/shm/params") if platform.system() != "Darwin" else self.params
    self.reset_params()
    self.scc_m = SmartCruiseControlMap()
    self.sm = MagicMock()
    self.sm.__getitem__.return_value = SimpleNamespace(latitude=0.0, longitude=0.0)

  def reset_params(self):
    self.params.put_bool("SmartCruiseControlMap", True, block=True)

    self.params.put("MapTargetVelocities", "{}", block=True)

  def test_initial_state(self):
    assert self.scc_m.state == VisionState.disabled
    assert not self.scc_m.is_active
    assert self.scc_m.output_v_target == V_CRUISE_UNSET
    assert self.scc_m.output_a_target == 0.

  def test_system_disabled(self):
    self.params.put_bool("SmartCruiseControlMap", False, block=True)
    self.scc_m.enabled = self.params.get_bool("SmartCruiseControlMap")

    for _ in range(int(10. / DT_MDL)):
      self.scc_m.update(self.sm, True, False, 0., 0., 0.)
    assert self.scc_m.state == VisionState.disabled
    assert not self.scc_m.is_active

  def test_disabled(self):
    for _ in range(int(10. / DT_MDL)):
      self.scc_m.update(self.sm, False, False, 0., 0., 0.)
    assert self.scc_m.state == VisionState.disabled

  def test_transition_disabled_to_enabled(self):
    for _ in range(int(10. / DT_MDL)):
      self.scc_m.update(self.sm, True, False, 0., 0., 0.)
    assert self.scc_m.state == VisionState.enabled

  def test_moderate_curve(self):
    # Regression: `... / 2 * a` parsed as `(.../2)*a` instead of `.../(2*a)`,
    # making max_d ~11x too small so the moderate-curve branch never tripped.
    # v_ego=25, a_ego=0, tv=24: fixed max_d≈45m vs buggy ≈4m at a 40m waypoint.
    waypoint_lon_deg = (40.0 / R) * (180.0 / math.pi)
    self.mem_params.put("MapTargetVelocities",
                        json.dumps([{"latitude": 0.0, "longitude": waypoint_lon_deg, "velocity": 24.0}]), block=True)

    self.scc_m.update(self.sm, True, False, 25.0, 0.0, 30.0)

    self.assertAlmostEqual(self.scc_m.v_target, 24.0, delta=24.0 * 1e-6)

  def test_limo_mode_holds_then_forgets_target(self):
    target_lon = (200.0 / R) * (180.0 / math.pi)
    self.mem_params.put("MapTargetVelocities", json.dumps([{"latitude": 0.0, "longitude": target_lon, "velocity": 15.0}]), block=True)
    self.scc_m.limo = SimpleNamespace(enabled=True, approach_distance=lambda v_target: 250.0)

    def drive_to(x, v_ego):
      self.sm.recv_time = {self.scc_m.gps_location_service: time.monotonic()}
      self.sm.__getitem__.return_value = SimpleNamespace(latitude=0.0, longitude=(x / R) * (180.0 / math.pi))
      self.scc_m.update(self.sm, True, False, v_ego, 0.0, 30.0)

    drive_to(0.0, 25.0)
    assert self.scc_m.v_target == 15.0
    # slower than the target before its place
    drive_to(150.0, 14.5)
    assert self.scc_m.v_target == 15.0
    # the map data is stuck, it has to let go once it's passed
    self.scc_m.limo.approach_distance = lambda v_target: 0.0
    for x in range(200, 260, 5):
      drive_to(float(x), 14.5)
    drive_to(260.0, 14.5)
    assert self.scc_m.v_target != 15.0

  @parameterized.expand([(0.0, False), (0.9, True), (100.0, True)])
  def test_limo_mode_position_is_old(self, gps_age, expected):
    # 25 m/s, 20 m short of the distance to start at
    target_lon = (285.0 / R) * (180.0 / math.pi)
    self.mem_params.put("MapTargetVelocities", json.dumps([{"latitude": 0.0, "longitude": target_lon, "velocity": 15.0}]), block=True)
    self.scc_m.limo = SimpleNamespace(enabled=True, approach_distance=lambda v_target: 250.0)
    self.sm.__getitem__.return_value = SimpleNamespace(latitude=0.0, longitude=0.0)
    self.sm.recv_time = {self.scc_m.gps_location_service: time.monotonic() - gps_age}
    self.scc_m.update(self.sm, True, False, 25.0, 0.0, 30.0)
    assert (self.scc_m.v_target == 15.0) == expected

  # TODO-SP: mock data from modelV2 to test other states
