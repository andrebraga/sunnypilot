#!/usr/bin/env python3
import numpy as np

import openpilot.cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.accel_boost import AccelBoost
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource, STOP_DISTANCE
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, should_stop
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
from openpilot.sunnypilot.selfdrive.controls.lib.creep_up import CreepUp

J_CRUISE_VALS = [1.6, 1.2, 0.8, 0.6]
J_CRUISE_BP = [0., 10.0, 25., 40.]
A_CRUISE_MIN = -0.5
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]


def get_cruise_accel(v_cruise, v_ego, a_cruise_prev, dt):
  target_accel = np.clip(v_cruise - v_ego, A_CRUISE_MIN, ACCEL_MAX)
  j_cruise = np.interp(v_ego, J_CRUISE_BP, J_CRUISE_VALS)
  target_accel = float(np.clip(target_accel, a_cruise_prev - j_cruise * dt, a_cruise_prev + j_cruise * dt))

  return target_accel


ANTICIPATION_HORIZON = 4.0  # s


class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, CP_SP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.accel_boost = AccelBoost()
    # Limo mode: a smoother lead plan with more room. It can only add braking to the lead plan, never take it away
    self.mpc_anticipation = LongitudinalMpc(dt=dt)
    self.creep_up = CreepUp(dt)
    self.creep_up_failed = False
    self.anticipating = False
    self.anticipation_failed = False

    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.a_cruise = init_a
    self.output_a_target = init_a
    self.output_should_stop = False

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  def update(self, sm):
    LongitudinalPlannerSP.update(self, sm)

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    if sm['controlsState'].forceDecel:
      v_cruise = 0.0

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET
    reset_state = reset_state or not v_cruise_initialized

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.output_a_target = np.clip(sm['carState'].aEgo, ACCEL_MIN, ACCEL_MAX)
      self.a_cruise = self.output_a_target

    limo_active = self.limo.enabled and not reset_state
    if not limo_active:
      # switching it on and off continues from the last target
      if self.limo.active:
        self.a_cruise = min(self.a_cruise, self.output_a_target)
      self.limo.reset(float(self.output_a_target))
    self.limo.start_frame(v_ego, float(np.clip(sm['carState'].aEgo, ACCEL_MIN, ACCEL_MAX)), sm['selfdriveState'].personality, limo_active)
    if limo_active:
      self.output_a_target = self.limo.a_plan

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    # Get new v_cruise and a_target from Smart Cruise Control and Speed Limit Assist
    v_cruise, self.output_a_target = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.output_a_target, v_cruise)

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.close_in = self.mpc_anticipation.close_in = not sm['carState'].standstill or self.creep_up.active
    if self.CP.openpilotLongitudinalControl:
      self.mpc.set_cur_state(self.v_desired_filter.x, self.output_a_target)
    else:
      self.mpc.set_cur_state(v_ego, sm['carState'].aEgo)
    self.mpc.update(sm['radarState'], personality=sm['selfdriveState'].personality)
    self.update_anticipation(sm, limo_active, prev_accel_constraint)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Save starting point for next iteration
    a_prev = self.output_a_target

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                              action_t=action_t)
    output_should_stop_mpc = should_stop(v_ego, output_a_target_mpc)
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    self.a_cruise = get_cruise_accel(v_cruise, v_ego, self.a_cruise, self.dt)
    cruise_should_stop = should_stop(v_ego, self.a_cruise)
    cruise_target = self.a_cruise
    limo_target = None
    if limo_active:
      # boost and starting need to know what cruise wants, not how far the smoothing got
      limo_target = self.limo.cruise_target(v_cruise, self.source)
      if limo_target is not None:
        cruise_target = limo_target
        cruise_should_stop = should_stop(v_ego, cruise_target)

    output_a_target_e2e = self.accel_boost.update(sm, output_a_target_e2e, output_a_target_mpc, cruise_target)

    candidates = [(output_a_target_mpc, self.mpc.source, output_should_stop_mpc),
                  (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop),
                  (output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e)]

    limo_plan = self.get_limo_plan(sm, cruise_target, output_a_target_mpc, output_a_target_e2e, action_t) if limo_target is not None else None
    if limo_plan is not None:
      limo_a_target, limo_a_cruise = limo_plan
      candidates[1] = (limo_a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop)

    output_a_target, self.mpc.source, _ = min(candidates, key=lambda c: c[0])
    self.output_should_stop = any(should_stop for _, _, should_stop in candidates)
    if limo_plan is not None:
      output_a_target = limo_a_target
      a_prev = self.limo.output_last
    elif limo_active:
      limo_a_target = self.limo.stay_out(float(np.clip(output_a_target, ACCEL_MIN, ACCEL_MAX)))
      if limo_a_target is not None:
        output_a_target = limo_a_target
    self.output_a_target = np.clip(output_a_target, ACCEL_MIN, ACCEL_MAX)
    self.update_creep_up(sm, v_ego, reset_state)

    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.output_a_target + a_prev) / 2.0

  def update_creep_up(self, sm, v_ego: float, reset_state: bool) -> None:
    if self.creep_up_failed:
      return
    try:
      CS = sm['carState']
      gap = self.mpc.stop_distance if self.mpc.stop_distance < STOP_DISTANCE else np.nan
      creep = self.creep_up.update(v_ego, CS.standstill, float(self.output_a_target), sm['radarState'].leadOne, gap,
                                   not reset_state, self.fcw, CS.gasPressed or CS.brakePressed)
      if creep is not None:
        self.output_a_target = float(np.clip(creep[0], ACCEL_MIN, ACCEL_MAX))
        self.output_should_stop = creep[1]
    except Exception:
      cloudlog.exception("creep up failed, off until restart")
      self.creep_up_failed = True

  def update_anticipation(self, sm, limo_active: bool, prev_accel_constraint: bool) -> None:
    curve = self.limo.curve
    anticipating = limo_active and curve.anticipation > 0.0 and not self.anticipation_failed
    try:
      if anticipating:
        mpc = self.mpc_anticipation
        if not self.anticipating:
          mpc.reset()
          mpc.a_prev[:] = self.output_a_target
        personality = sm['selfdriveState'].personality
        mpc.set_weights(prev_accel_constraint, personality=personality, jerk_scale=curve.jerk_scale)
        mpc.set_cur_state(self.v_desired_filter.x, self.output_a_target)
        mpc.update(sm['radarState'], personality=personality, extra_follow=curve.extra_follow)
        # a solver that failed this time has reset its plan to standing still
        anticipating = bool(abs(mpc.v_solution[0] - self.v_desired_filter.x) < 0.1 and np.isfinite(mpc.a_solution).all())
    except Exception:
      cloudlog.exception("limo mode anticipation failed, off until restart")
      self.anticipation_failed = True
      anticipating = False
    self.anticipating = anticipating

  def get_anticipation(self, action_t: float) -> float:
    """The smoother lead plan's target, inf when there isn't one"""
    if not self.anticipating:
      return np.inf
    v = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc_anticipation.v_solution)
    a = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc_anticipation.a_solution)
    target = get_accel_from_plan(v, a, CONTROL_N_T_IDX, action_t=action_t)
    # earlier, not more: never braking harder than the lead plan itself does within the next few seconds
    ahead = self.mpc.a_solution[(T_IDXS_MPC >= action_t) & (T_IDXS_MPC <= action_t + ANTICIPATION_HORIZON)]
    target = max(target, float(np.min(ahead))) if len(ahead) else np.inf
    return target if np.isfinite(target) else np.inf

  def get_limo_plan(self, sm, cruise_target, a_lead, a_model, action_t):
    a_lead_ahead = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                       action_t=action_t + self.limo.lookahead)
    a_lead = self.limo.anticipate(a_lead, self.get_anticipation(action_t))
    a_lead_ahead = self.limo.anticipate(a_lead_ahead, self.get_anticipation(action_t + self.limo.lookahead))
    a_model_ahead = a_model
    model_v, model_a = sm['modelV2'].velocity.x, sm['modelV2'].acceleration.x
    if len(model_v) == len(model_a) == len(ModelConstants.T_IDXS):
      a_now, a_ahead = (get_accel_from_plan(model_v, model_a, ModelConstants.T_IDXS, action_t=t)
                        for t in (action_t, action_t + self.limo.lookahead))
      a_model_ahead = self.limo.model_accel(a_model, a_now, a_ahead)

    leads = (sm['radarState'].leadOne, sm['radarState'].leadTwo)
    lead_distance = min((lead.dRel for lead in leads if lead.present), default=np.inf)
    return self.limo.plan(cruise_target, a_lead, a_model, a_lead_ahead, a_model_ahead, lead_distance, self.fcw)

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks()

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.present
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.accelBoost = float(self.accel_boost.total_boost)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm)
