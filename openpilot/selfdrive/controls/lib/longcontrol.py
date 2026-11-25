import numpy as np
from opendbc.car.structs import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.common.pid import PIDController
from openpilot.selfdrive.modeld.constants import ModelConstants

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

LongCtrlState = car.CarControl.Actuators.LongControlState

# Chauffeur stop (Limo mode): let the brakes go as the car comes to rest, like i-PEDAL does, and only tighten them for
# the hold once it stands. Braking that rises into the halt is what makes the body bob back after it
SOFT_STOP_GAIN = 1.5  # 1/s, braking per speed near rest, about what i-PEDAL does
SOFT_STOP_FLOOR = 0.15  # m/s^2, so it does come to rest
SOFT_STOP_JERK = 1.0  # m/s^3, how quickly it follows that, the same as the stock squeeze
SOFT_STOP_TIME = 2.0  # s, still rolling after this: squeeze as without it
SOFT_STOP_SETTLE = 0.5  # s, standstill shows a little before the car really stands
SOFT_STOP_SPEEDUP = 0.05  # m/s, rolling faster than when it started: squeeze as without it


def long_control_state_trans(CP_SP, active, long_control_state,
                             should_stop, brake_pressed, cruise_standstill):
  # Gas Interceptor
  cruise_standstill = cruise_standstill and not CP_SP.enableGasInterceptor

  starting_condition = (not should_stop and
                        not cruise_standstill and
                        not brake_pressed)

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if not starting_condition:
        long_control_state = LongCtrlState.stopping
      else:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.pid:
      if should_stop:
        long_control_state = LongCtrlState.stopping

  return long_control_state

class LongControl:
  def __init__(self, CP, CP_SP):
    self.CP = CP
    self.CP_SP = CP_SP
    self.long_control_state = LongCtrlState.off
    self.pid = PIDController(0.0, (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             rate=1 / DT_CTRL)
    self.last_output_accel = 0.0
    self.soft_stop_frames = 0
    self.soft_standstill_frames = 0
    self.soft_stop_v = 0.0
    self.soft_stopping = False

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, a_target, should_stop, accel_limits, soft_stop=False):
    """Update longitudinal control. This updates the state machine and runs a PID loop

    soft_stop: the chauffeur stop may be used (Limo mode on, no FCW, no lead close)
    """
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    self.long_control_state = long_control_state_trans(self.CP_SP, active, self.long_control_state,
                                                       should_stop, CS.brakePressed,
                                                       CS.cruiseState.standstill)
    if self.long_control_state != LongCtrlState.stopping:
      # only for stops that start while rolling, and once per stop: anything that ends it doesn't come back
      self.soft_stopping = soft_stop and not CS.standstill
      self.soft_stop_frames = 0
      self.soft_standstill_frames = 0
      self.soft_stop_v = CS.vEgo
    elif self.soft_stopping:
      self.soft_stop_frames += 1
      self.soft_standstill_frames = self.soft_standstill_frames + 1 if CS.standstill else 0
      self.soft_stopping = soft_stop and self.soft_stop_frames * DT_CTRL < SOFT_STOP_TIME and \
                           self.soft_standstill_frames * DT_CTRL < SOFT_STOP_SETTLE and \
                           CS.vEgo <= self.soft_stop_v + SOFT_STOP_SPEEDUP

    if self.long_control_state == LongCtrlState.off:
      self.reset()
      output_accel = 0.

    elif self.long_control_state == LongCtrlState.stopping and self.soft_stopping:
      # never less than the plan asks, just not the extra squeeze before the car stands
      target = min(-max(SOFT_STOP_FLOOR, SOFT_STOP_GAIN * CS.vEgo), a_target)
      output_accel = min(self.last_output_accel, 0.0)
      step = SOFT_STOP_JERK * DT_CTRL
      output_accel += float(np.clip(target - output_accel, -step, step))
      self.reset()

    elif self.long_control_state == LongCtrlState.stopping:
      output_accel = self.last_output_accel
      if output_accel > self.CP.stopAccel:
        output_accel = min(output_accel, 0.0)
        # TODO: can we just go straight to stopAccel?
        output_accel -= 1.0 * DT_CTRL  # m/s^2/s while trying to stop
      self.reset()

    else:  # LongCtrlState.pid
      error = a_target - CS.aEgo
      output_accel = self.pid.update(error, speed=CS.vEgo,
                                     feedforward=a_target)

    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel
