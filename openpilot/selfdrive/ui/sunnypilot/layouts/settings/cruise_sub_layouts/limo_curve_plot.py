"""
Copyright (c) 2021-, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
import threading
from collections.abc import Callable

import numpy as np
import pyray as rl

from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.selfdrive.controls.lib.limo_curve import LimoCurve
from openpilot.sunnypilot.selfdrive.controls.lib.limo_mode import preview
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

DURATION = 30.0  # s, the weakest curve takes this long
V_TARGET = 50.0 * CV.KPH_TO_MS
V_TARGET_MPH = 30.0 * CV.MPH_TO_MS
T_MIN = 10.0  # s, shortest time axis

INK = rl.Color(255, 255, 255, 255)
INK_SECONDARY = rl.Color(195, 194, 190, 255)
INK_MUTED = rl.Color(140, 139, 134, 255)
GRID = rl.Color(44, 44, 42, 255)
AXIS = rl.Color(82, 81, 78, 255)
SERIES = rl.Color(0x39, 0x87, 0xE5, 255)
REFERENCE = rl.Color(140, 139, 134, 255)
TITLE_SIZE = 32
TICK_SIZE = 28


class Launch:
  """What a curve does to a launch on an open road"""
  def __init__(self, curve: LimoCurve, v_target: float):
    self.curve = curve
    self.v_target = v_target
    self.speed, self.accel = preview(curve, v_target, DURATION)
    self.t = np.arange(len(self.speed)) * DT_MDL
    there = np.flatnonzero(self.speed > 0.97 * v_target)
    self.time = float(self.t[there[0]]) if len(there) else DURATION
    self.strength = float(self.accel.max())
    self.jerk = float(np.abs(np.diff(self.accel)).max() / DT_MDL)


class LaunchWorker:
  """Computes launches off the UI thread, always the latest one asked for"""
  def __init__(self):
    self._lock = threading.Lock()
    self._wanted: tuple[LimoCurve, float] | None = None
    self._running = False
    self._cache: dict[tuple[LimoCurve, float], Launch] = {}

  def get(self, curve: LimoCurve, v_target: float) -> Launch | None:
    key = (curve, v_target)
    with self._lock:
      if key in self._cache:
        return self._cache[key]
      self._wanted = key
      if not self._running:
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()
    return None

  def _run(self) -> None:
    while True:
      with self._lock:
        key, self._wanted = self._wanted, None
        if key is None:
          self._running = False
          return
      try:
        launch = Launch(*key)
      except Exception:
        cloudlog.exception("limo curve preview failed")
        with self._lock:
          self._running = False
        return
      with self._lock:
        if len(self._cache) > 40:
          self._cache.clear()
        self._cache[key] = launch


class LimoCurvePlot(Widget):
  """The acceleration and the speed of a launch, with the edited curve and the default one"""
  def __init__(self, curve: Callable[[], LimoCurve], default: Callable[[], LimoCurve], is_metric: Callable[[], bool]):
    super().__init__()
    self._curve, self._default, self._is_metric = curve, default, is_metric
    self._curves = LaunchWorker()
    self._defaults = LaunchWorker()
    self._launch: Launch | None = None
    self._reference: Launch | None = None
    self._t_max = T_MIN
    self._font = gui_app.font(FontWeight.MEDIUM)
    self._font_tick = gui_app.font(FontWeight.NORMAL)

  def _update_state(self) -> None:
    super()._update_state()
    v_target = V_TARGET if self._is_metric() else V_TARGET_MPH
    # keep showing the last one until it's there
    self._launch = self._curves.get(self._curve(), v_target) or self._launch
    self._reference = self._defaults.get(self._default(), v_target) or self._reference
    launches = [launch for launch in (self._launch, self._reference) if launch is not None]
    if launches:
      self._t_max = min(max(T_MIN, 5.0 * math.ceil((max(launch.time for launch in launches) + 1.0) / 5.0)), DURATION)

  def _text(self, text: str, x: float, y: float, size: int, color: rl.Color, align: float = 0.0, medium: bool = False) -> float:
    font = self._font if medium else self._font_tick
    width = measure_text_cached(font, text, size).x
    rl.draw_text_ex(font, text, rl.Vector2(x - align * width, y), size, 0, color)
    return width

  def _render(self, rect: rl.Rectangle) -> None:
    metric = self._is_metric()
    unit, scale, target = ("km/h", CV.MS_TO_KPH, 50) if metric else ("mph", CV.MS_TO_MPH, 30)

    gap = 60
    stats_height = 140
    width = (rect.width - gap) / 2
    height = rect.height - stats_height
    self._render_stats(rl.Rectangle(rect.x, rect.y, rect.width, stats_height), f"0–{target} {unit}")
    y = rect.y + stats_height
    self._render_chart(rl.Rectangle(rect.x, y, width, height), tr("Acceleration") + ", m/s^2", lambda launch: launch.accel, (0.0, 1.0, 2.0), 2.2)
    self._render_chart(rl.Rectangle(rect.x + width + gap, y, width, height), tr("Speed") + f", {unit}", lambda launch: launch.speed * scale,
                       tuple(target * f for f in (0.0, 0.5, 1.0)), target * 1.1)

  def _render_stats(self, rect: rl.Rectangle, launch_name: str) -> None:
    a, b = self._launch, self._reference
    if a is None or b is None:
      return
    stats = ((launch_name, f"{a.time:.1f} s", f"{b.time:.1f}"),
             (tr("Strongest push"), f"{a.strength:.2f} m/s^2", f"{b.strength:.2f}"),
             (tr("Sharpest change"), f"{a.jerk:.1f} m/s^3", f"{b.jerk:.1f}"))
    legend_width = 300
    width = (rect.width - legend_width) / len(stats)
    for i, (name, value, reference) in enumerate(stats):
      x = rect.x + i * width
      self._text(name, x, rect.y, 30, INK_SECONDARY)
      self._text(value, x, rect.y + 34, 46, INK, medium=True)
      self._text(tr("default") + " " + reference, x, rect.y + 88, TICK_SIZE, INK_MUTED)

    x = rect.x + rect.width - legend_width + 40
    for i, (name, color, dashed) in enumerate(((tr("This curve"), SERIES, False), (tr("Default"), REFERENCE, True))):
      y = rect.y + 18 + i * 42
      self._line([(x, y), (x + 60, y)], color, dashed)
      self._text(name, x + 76, y - 16, 30, INK_SECONDARY)

  def _render_chart(self, rect: rl.Rectangle, title: str, values: Callable[[Launch], np.ndarray], ticks: tuple[float, ...], y_max: float) -> None:
    self._text(title, rect.x, rect.y, TITLE_SIZE, INK_SECONDARY)
    plot = rl.Rectangle(rect.x + 56, rect.y + 52, rect.width - 56 - 24, rect.height - 52 - 44)

    def to_xy(t, v):
      return plot.x + t / self._t_max * plot.width, plot.y + plot.height * (1.0 - v / y_max)

    for tick in ticks:
      _, y = to_xy(0.0, tick)
      rl.draw_line_ex(rl.Vector2(plot.x, y), rl.Vector2(plot.x + plot.width, y), 2, AXIS if tick == 0.0 else GRID)
      self._text(f"{tick:g}", plot.x - 14, y - TICK_SIZE / 2, TICK_SIZE, INK_MUTED, align=1.0)
    for i in range(int(self._t_max // 5) + 1):
      x, y = to_xy(5.0 * i, 0.0)
      self._text(f"{5 * i} s", x, y + 8, TICK_SIZE, INK_MUTED, align=0.5 if i else 0.0)

    for launch, color, dashed in ((self._reference, REFERENCE, True), (self._launch, SERIES, False)):
      if launch is None:
        continue
      n = int(self._t_max / DT_MDL) + 1
      self._line([to_xy(t, v) for t, v in zip(launch.t[:n:2], values(launch)[:n:2], strict=True)], color, dashed)

  @staticmethod
  def _line(points: list[tuple[float, float]], color: rl.Color, dashed: bool) -> None:
    width = 4 if dashed else 6
    if not dashed:
      for p, q in zip(points, points[1:], strict=False):
        rl.draw_line_ex(rl.Vector2(*p), rl.Vector2(*q), width, color)
      return
    # dashes of the same length along the line
    on, off, along = 14.0, 10.0, 0.0
    for (px, py), (qx, qy) in zip(points, points[1:], strict=False):
      length = math.hypot(qx - px, qy - py)
      s = 0.0
      while s < length:
        phase = (along + s) % (on + off)
        step = min((on if phase < on else on + off) - phase, length - s)
        if phase < on and step > 0.0:
          a, b = s / length, (s + step) / length
          rl.draw_line_ex(rl.Vector2(px + (qx - px) * a, py + (qy - py) * a), rl.Vector2(px + (qx - px) * b, py + (qy - py) * b), width, color)
        s += max(step, 1e-3)
      along += length
