"""
Copyright (c) 2021-, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from collections.abc import Callable

import pyray as rl

from openpilot.system.ui.lib.application import FontWeight, MouseEvent, MousePos, gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.sunnypilot.lib.styles import style
from openpilot.system.ui.widgets import Widget

BUTTON_SIZE = 110
SPACING = 30
TRACK_HEIGHT = 14
KNOB_RADIUS = 30
KNOB_RADIUS_PRESSED = 38
TITLE_FONT_SIZE = 44
VALUE_FONT_SIZE = 44
BUTTON_FONT_SIZE = 60
REPEAT_DELAY = 0.45  # s
REPEAT_INTERVAL = 0.09  # s


class SliderStepper(Widget):
  """Slider for the coarse value, buttons on both ends to refine it. The buttons repeat when held"""

  def __init__(self, title: str | Callable[[], str], min_value: float, max_value: float, step: float, value: float,
               on_changed: Callable[[float], None], on_released: Callable[[], None] | None = None,
               label: Callable[[float], str] = lambda v: f"{v:.2f}", hints: tuple[str, str] = ("", "")):
    super().__init__()
    self._title = title
    self._min, self._max, self._step = min_value, max_value, step
    self._n_steps = int(round((max_value - min_value) / step))
    self._value = min_value
    self._default: float | None = None
    self._on_changed = on_changed
    self._on_released = on_released
    self._label = label
    self._hints = hints
    self._dragging = False
    self._held = 0  # direction of the held button
    self._repeat_at = 0.0
    self._font = gui_app.font(FontWeight.MEDIUM)
    self._font_hint = gui_app.font(FontWeight.NORMAL)
    self._minus_rect = self._plus_rect = self._track_rect = rl.Rectangle(0, 0, 0, 0)
    self.set_value(value)

  @property
  def value(self) -> float:
    return self._value

  @property
  def active(self) -> bool:
    return self._dragging or self._held != 0

  def set_value(self, value: float) -> None:
    """Does not call back"""
    self._value = self._snap(value)

  def set_default(self, value: float) -> None:
    """Marked on the track"""
    self._default = value

  def _snap(self, value: float) -> float:
    index = int(round((value - self._min) / self._step))
    return round(self._min + min(max(index, 0), self._n_steps) * self._step, 6)

  def _change(self, value: float) -> None:
    value = self._snap(value)
    if value != self._value:
      self._value = value
      self._on_changed(value)

  def _fraction(self, value: float) -> float:
    return min(max((value - self._min) / (self._max - self._min), 0.0), 1.0)

  def _drag_to(self, x: float) -> None:
    fraction = (x - self._track_rect.x) / max(self._track_rect.width, 1.0)
    self._change(self._min + min(max(fraction, 0.0), 1.0) * (self._max - self._min))

  def _update_layout_rects(self) -> None:
    r = self._rect
    y = r.y + r.height - BUTTON_SIZE
    self._minus_rect = rl.Rectangle(r.x, y, BUTTON_SIZE, BUTTON_SIZE)
    self._plus_rect = rl.Rectangle(r.x + r.width - BUTTON_SIZE, y, BUTTON_SIZE, BUTTON_SIZE)
    x = self._minus_rect.x + BUTTON_SIZE + SPACING + KNOB_RADIUS
    width = self._plus_rect.x - SPACING - KNOB_RADIUS - x
    self._track_rect = rl.Rectangle(x, y + (BUTTON_SIZE - TRACK_HEIGHT) / 2, width, TRACK_HEIGHT)

  def _handle_mouse_press(self, mouse_pos: MousePos) -> None:
    if rl.check_collision_point_rec(mouse_pos, self._minus_rect):
      self._hold(-1)
    elif rl.check_collision_point_rec(mouse_pos, self._plus_rect):
      self._hold(1)
    elif self._minus_rect.x + BUTTON_SIZE < mouse_pos.x < self._plus_rect.x and mouse_pos.y >= self._minus_rect.y:
      self._dragging = True
      self._drag_to(mouse_pos.x)

  def _hold(self, direction: int) -> None:
    self._held = direction
    self._repeat_at = rl.get_time() + REPEAT_DELAY
    self._change(self._value + direction * self._step)

  def _handle_mouse_event(self, mouse_event: MouseEvent) -> None:
    if mouse_event.slot != 0:
      return
    if mouse_event.left_released:
      self._release()
    elif mouse_event.left_down and self._dragging:
      self._drag_to(mouse_event.pos.x)

  def _release(self) -> None:
    if self.active:
      self._dragging, self._held = False, 0
      if self._on_released is not None:
        self._on_released()

  def hide_event(self) -> None:
    super().hide_event()
    self._release()

  def _update_state(self) -> None:
    super()._update_state()
    # the release can get lost, for example when the screen turns off
    if self.active and (not self.enabled or gui_app.last_mouse_event.left_released or not self._touch_valid()):
      self._release()
    if self._held != 0 and rl.get_time() >= self._repeat_at:
      self._repeat_at = rl.get_time() + REPEAT_INTERVAL
      self._change(self._value + self._held * self._step)

  def _render(self, _) -> None:
    text_color = style.ITEM_TEXT_COLOR if self.enabled else style.ITEM_DISABLED_TEXT_COLOR
    title = self._title() if callable(self._title) else self._title
    title_size = measure_text_cached(self._font, title, TITLE_FONT_SIZE)
    title_y = self._minus_rect.y - title_size.y - 2
    rl.draw_text_ex(self._font, title, rl.Vector2(self._rect.x, title_y), TITLE_FONT_SIZE, 0, text_color)

    value = self._label(self._value)
    value_size = measure_text_cached(self._font, value, VALUE_FONT_SIZE)
    rl.draw_text_ex(self._font, value, rl.Vector2(self._rect.x + self._rect.width - value_size.x, title_y), VALUE_FONT_SIZE, 0, text_color)

    self._render_button(self._minus_rect, "-", self._value > self._min, self._held < 0)
    self._render_button(self._plus_rect, "+", self._value < self._max, self._held > 0)

    track = self._track_rect
    knob_x = track.x + self._fraction(self._value) * track.width
    center_y = track.y + track.height / 2
    filled = style.ON_BG_COLOR if self.enabled else style.DISABLED_ON_BG_COLOR
    rl.draw_rectangle_rounded(track, 1.0, 10, style.OFF_BG_COLOR)
    rl.draw_rectangle_rounded(rl.Rectangle(track.x, track.y, max(knob_x - track.x, TRACK_HEIGHT), track.height), 1.0, 10, filled)
    if self._default is not None:
      x = track.x + self._fraction(self._default) * track.width
      rl.draw_rectangle_rounded(rl.Rectangle(x - 3, center_y - 22, 6, 44), 1.0, 6, style.ITEM_DESC_TEXT_COLOR)
    rl.draw_circle_v(rl.Vector2(knob_x, center_y), KNOB_RADIUS_PRESSED if self._dragging else KNOB_RADIUS,
                     style.TOGGLE_KNOB_COLOR if self.enabled else style.TOGGLE_DISABLED_KNOB_COLOR)

    for text, right in zip(self._hints, (False, True), strict=True):
      if text:
        size = measure_text_cached(self._font_hint, text, 28)
        x = track.x + track.width + KNOB_RADIUS - size.x if right else track.x - KNOB_RADIUS
        rl.draw_text_ex(self._font_hint, text, rl.Vector2(x, self._rect.y + self._rect.height - size.y), 28, 0, style.ITEM_DESC_TEXT_COLOR)

  def _render_button(self, rect: rl.Rectangle, text: str, enabled: bool, held: bool) -> None:
    enabled = enabled and self.enabled
    background = style.OPTION_CONTROL_BTN_PRESSED if held and enabled else style.OPTION_CONTROL_CONTAINER_BG
    rl.draw_rectangle_rounded(rect, 0.25, 12, background if enabled else style.OPTION_CONTROL_BTN_DISABLED)
    size = measure_text_cached(self._font, text, BUTTON_FONT_SIZE)
    rl.draw_text_ex(self._font, text, rl.Vector2(rect.x + (rect.width - size.x) / 2, rect.y + (rect.height - size.y) / 2), BUTTON_FONT_SIZE, 0,
                    style.ITEM_TEXT_COLOR if enabled else style.ITEM_DISABLED_TEXT_COLOR)
