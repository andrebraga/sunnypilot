from types import SimpleNamespace

from opendbc.car import structs
from opendbc.car.structs import car
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP
from openpilot.cereal import custom
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.radard import RADAR_TO_CAMERA, KalmanParams, Track, get_lead, radar_point_eligible_for_fusion


def radar_point(motion_state: int):
  point = car.RadarData.RadarPoint.new_message()
  point.motionState = motion_state
  return point


def test_hyundai_full_radar_fuses_only_classified_points():
  CP = structs.CarParams(brand="hyundai")
  CP_SP = custom.CarParamsSP.new_message()
  CP_SP.flags = HyundaiFlagsSP.RADAR_FULL_RADAR.value

  assert radar_point_eligible_for_fusion(CP, CP_SP, radar_point(1))
  assert radar_point_eligible_for_fusion(CP, CP_SP, radar_point(2))
  assert not radar_point_eligible_for_fusion(CP, CP_SP, radar_point(0))
  assert not radar_point_eligible_for_fusion(CP, CP_SP, radar_point(255))


def test_radar_motion_filter_does_not_affect_other_modes_or_brands():
  CP_SP = custom.CarParamsSP.new_message()

  assert radar_point_eligible_for_fusion(structs.CarParams(brand="hyundai"), CP_SP, radar_point(0))
  assert radar_point_eligible_for_fusion(structs.CarParams(brand="toyota"), CP_SP, radar_point(0))


V_EGO = 10.0


def model_lead(d: float, v: float):
  return SimpleNamespace(x=[d + RADAR_TO_CAMERA], y=[0.0], v=[v], a=[0.0], xStd=[3.0], yStd=[0.5], vStd=[1.0])


def track(identifier: int, d: float, v: float, stationary: bool, y: float = 0.0, v_ego: float = V_EGO):
  t = Track(identifier, v, KalmanParams(DT_MDL))
  t.update(d, y, v - v_ego, v, stationary)
  return t


def lead(tracks, model, prob=0.9, v_ego=V_EGO):
  CP = structs.CarParams(brand="hyundai")
  return get_lead(v_ego, True, {t.identifier: t for t in tracks}, model, v_ego, prob, CP, custom.CarParamsSP.new_message())


def test_stationary_track_is_not_a_moving_lead():
  # next to the lead, and the only or the best match
  out = lead([track(1, 18.0, 0.0, True)], model_lead(18.0, 9.8))
  assert out["present"] and not out["radar"]
  assert abs(out["dRel"] - 18.0) < 1e-3 and abs(out["vLead"] - 9.8) < 1e-3

  out = lead([track(1, 18.0, 0.0, True), track(2, 20.0, 9.5, False)], model_lead(18.0, 9.8))
  assert out["radar"] and out["radarTrackId"] == 2


def test_stationary_track_is_a_stopped_lead():
  for v_model in (0.0, 1.0, 2.5, 4.5):
    out = lead([track(1, 30.0, 0.0, True)], model_lead(31.0, v_model))
    assert out["radar"] and out["radarTrackId"] == 1


def test_stopping_lead_is_still_matched():
  # from moving to stationary, the model sees it too
  for v in (6.0, 4.0, 2.0, 0.5, 0.0):
    out = lead([track(1, 15.0, v, v < 1.0)], model_lead(15.0, v + 1.0))
    assert out["radar"] and out["radarTrackId"] == 1


def test_low_speed_override_works_for_stationary_tracks():
  model = model_lead(50.0, 0.0)
  for stationary in (False, True):
    out = lead([track(1, 3.0, 0.0, stationary, v_ego=3.0)], model, prob=0.0, v_ego=3.0)
    assert out["present"] and out["radar"]


def test_radars_without_motion_state_are_unaffected():
  out = lead([track(1, 18.0, 4.0, False)], model_lead(18.0, 9.8))
  assert out["radar"]
