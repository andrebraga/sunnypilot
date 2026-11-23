"""
Copyright (c) 2021-, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from collections.abc import Callable
from dataclasses import replace

import pyray as rl

from openpilot.selfdrive.ui.sunnypilot.layouts.settings.cruise_sub_layouts.limo_curve_plot import LimoCurvePlot
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.selfdrive.controls.lib.limo_curve import DEFAULTS, PARAM, PERSONALITIES, RANGES, LimoCurve, curves_from_param, \
  curves_to_param
from openpilot.system.ui.lib.application import MousePos
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import ButtonSP, MultipleButtonActionSP
from openpilot.system.ui.sunnypilot.widgets.slider_stepper import SliderStepper
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import ButtonStyle
from openpilot.system.ui.widgets.network import NavButton

HEADER_HEIGHT = 120
SLIDER_HEIGHT = 172
SLIDER_SPACING = 14
BACK_WIDTH = 230
RESET_WIDTH = 240


class LimoCurveSettingsLayout(Widget):
  """The curve of each driving personality"""
  def __init__(self, back_btn_callback: Callable):
    super().__init__()
    self._curves = dict(DEFAULTS)
    self._before_reset: LimoCurve | None = None

    self._back_button = self._child(NavButton(tr("Back")))
    self._back_button.set_click_callback(back_btn_callback)

    self._personality = self._child(MultipleButtonActionSP([lambda: tr("Aggressive"), lambda: tr("Standard"), lambda: tr("Relaxed")],
                                                           270, callback=self._on_personality))
    self._reset_button = self._child(ButtonSP(lambda: tr("Undo") if self._before_reset is not None else tr("Reset"), self._on_reset,
                                              button_style=ButtonStyle.NORMAL, border_radius=20, font_size=44))
    self._plot = self._child(LimoCurvePlot(lambda: self.curve, lambda: DEFAULTS[self.name], lambda: ui_state.is_metric))

    def slider(key: str, title: Callable[[], str], label: Callable[[float], str], hints: tuple[str, str]) -> SliderStepper:
      r = RANGES[key]
      return self._child(SliderStepper(title, r.min, r.max, r.step, r.min, lambda value: self._on_changed(key, value), self._save, label, hints))

    self._sliders = {
      "strength": slider("strength", lambda: tr("Strength: how hard it accelerates at most"), lambda v: f"{v:.2f} m/s^2",
                         (tr("gentle"), tr("sporty"))),
      "build": slider("build", lambda: tr("Build-up: how long until it pushes"), lambda v: f"{v:.2f} s", (tr("immediate"), tr("gradual"))),
      "ease": slider("ease", lambda: tr("Ease-off: how long it takes to settle"), lambda v: f"{v:.1f} s", (tr("short"), tr("long"))),
    }

  @property
  def name(self) -> str:
    return PERSONALITIES[self._personality.selected_button]

  @property
  def curve(self) -> LimoCurve:
    return self._curves[self.name]

  def show_event(self) -> None:
    super().show_event()
    self._curves = curves_from_param(ui_state.params.get(PARAM))
    personality = ui_state.params.get("LongitudinalPersonality", return_default=True)
    self._personality.selected_button = min(max(int(personality), 0), len(PERSONALITIES) - 1)
    self._load()

  def _load(self) -> None:
    self._before_reset = None
    for key, slider in self._sliders.items():
      slider.set_value(getattr(self.curve, key))
      slider.set_default(getattr(DEFAULTS[self.name], key))

  def _save(self) -> None:
    ui_state.params.put(PARAM, curves_to_param(self._curves))

  def _on_personality(self, _: int) -> None:
    self._load()

  def _on_changed(self, key: str, value: float) -> None:
    self._before_reset = None
    self._curves[self.name] = replace(self.curve, **{key: value})

  def _on_reset(self) -> None:
    if self._before_reset is not None:
      curve, undo = self._before_reset, None
    else:
      curve, undo = DEFAULTS[self.name], self.curve
    self._curves[self.name] = curve
    self._load()
    self._before_reset = undo
    self._save()

  def _update_state(self) -> None:
    super()._update_state()
    self._reset_button.set_enabled(self._before_reset is not None or self.curve != DEFAULTS[self.name])

  def _handle_mouse_release(self, mouse_pos: MousePos) -> None:
    pass

  def _render(self, rect: rl.Rectangle) -> None:
    self._back_button.render(rl.Rectangle(rect.x, rect.y + 10, BACK_WIDTH, HEADER_HEIGHT - 20))
    personality_width = len(self._personality.buttons) * self._personality.button_width
    self._personality.render(rl.Rectangle(rect.x + (rect.width - personality_width) / 2, rect.y, personality_width, HEADER_HEIGHT))
    self._reset_button.render(rl.Rectangle(rect.x + rect.width - RESET_WIDTH, rect.y + 5, RESET_WIDTH, HEADER_HEIGHT - 10))

    y = rect.y + rect.height
    for slider in reversed(self._sliders.values()):
      y -= SLIDER_HEIGHT
      slider.render(rl.Rectangle(rect.x, y, rect.width, SLIDER_HEIGHT))
      y -= SLIDER_SPACING

    top = rect.y + HEADER_HEIGHT + 24
    self._plot.render(rl.Rectangle(rect.x, top, rect.width, y - top))
