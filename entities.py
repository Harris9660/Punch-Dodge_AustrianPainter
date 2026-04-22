from typing import Optional

import cv2
import numpy as np

from game_utils import Point, Rect, blend_circle, blend_rect
from settings import PAD_TARGET_IMAGE_PATH


def _load_pad_target_image():
    """Load the configured pad image once and reuse it across frames."""
    if PAD_TARGET_IMAGE_PATH is None:
        return None

    image = cv2.imread(PAD_TARGET_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        return None
    return image


class Attack:
    """Shared interface for attacks the player has to dodge."""

    def update(self, now: float, delta_time: float) -> None:
        raise NotImplementedError

    def draw(self, frame, now: float) -> None:
        raise NotImplementedError

    def get_hit_rect(self) -> Optional[Rect]:
        raise NotImplementedError

    def is_finished(self, frame_w: int, frame_h: int) -> bool:
        raise NotImplementedError


class JabAttack(Attack):
    """A jab grows toward the face before becoming dangerous."""

    def __init__(
        self,
        center: Point,
        start_radius: int,
        strike_radius: int,
        grow_duration: float,
        active_duration: float,
        start_time: float,
        label: str = "JAB",
    ):
        self.center = center
        self.start_radius = start_radius
        self.strike_radius = strike_radius
        self.grow_duration = grow_duration
        self.active_duration = active_duration
        self.activation_time = start_time
        self.label = label
        self.state = "queued"
        self.state_started_at = 0.0
        self.finished = False

    def get_current_radius(self, now: float) -> int:
        """Grow from the pad-work start size into the live jab size."""
        elapsed = max(0.0, now - self.state_started_at)
        if elapsed >= self.grow_duration:
            return self.strike_radius
        progress = elapsed / max(self.grow_duration, 0.001)
        radius = self.start_radius + (self.strike_radius - self.start_radius) * progress
        return int(round(radius))

    def update(self, now: float, delta_time: float) -> None:
        del delta_time
        if self.finished:
            return

        if self.state == "queued":
            if now < self.activation_time:
                return
            self.state = "warning"
            self.state_started_at = now
            return

        elapsed = now - self.state_started_at
        if self.state == "warning" and elapsed >= self.grow_duration:
            self.state = "strike"
            self.state_started_at = now
        elif self.state == "strike" and elapsed >= self.active_duration:
            self.finished = True

    def draw(self, frame, now: float) -> None:
        if self.state == "queued":
            return

        cx, cy = self.center

        if self.state == "warning":
            radius = self.get_current_radius(now)
            blend_circle(frame, self.center, radius, (0, 215, 255), 0.22)
            cv2.circle(frame, self.center, radius, (0, 215, 255), 3)
            cv2.circle(frame, self.center, max(5, radius // 4), (0, 215, 255), -1)

            remaining = max(0.0, self.grow_duration - (now - self.state_started_at))
            cv2.putText(
                frame,
                f"{self.label} IN {remaining:.1f}",
                (max(10, min(cx - 85, frame.shape[1] - 190)), max(30, cy - self.strike_radius - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 215, 255),
                2,
            )
        else:
            blend_circle(frame, self.center, self.strike_radius, (0, 0, 255), 0.45)
            cv2.circle(frame, self.center, self.strike_radius, (0, 0, 255), 3)
            cv2.circle(frame, self.center, max(5, self.strike_radius // 4), (0, 0, 255), -1)
            cv2.putText(
                frame,
                "MOVE!",
                (max(10, min(cx - 45, frame.shape[1] - 100)), max(30, cy - self.strike_radius - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

    def get_hit_rect(self) -> Optional[Rect]:
        if self.state == "strike" and not self.finished:
            cx, cy = self.center
            diameter = self.strike_radius * 2
            return cx - self.strike_radius, cy - self.strike_radius, diameter, diameter
        return None

    def is_finished(self, frame_w: int, frame_h: int) -> bool:
        del frame_w, frame_h
        if self.state == "queued":
            return False
        return self.finished


class HookAttack(Attack):
    """A hook flashes a warning beside the head before sweeping across the screen."""

    def __init__(
        self,
        spawn_side: str,
        speed: float,
        width: int,
        height: int,
        frame_w: int,
        y: int,
        warning_rect: Rect,
        warning_duration: float,
        start_time: float,
    ):
        self.spawn_side = spawn_side
        self.width = width
        self.height = height
        self.frame_w = frame_w
        self.y = y
        self.warning_rect = warning_rect
        self.activation_time = start_time
        self.warning_duration = warning_duration
        self.state = "queued"
        self.state_started_at = 0.0

        if spawn_side == "left":
            self.x = -width
            self.vx = speed
        elif spawn_side == "right":
            self.x = frame_w
            self.vx = -speed
        else:
            raise ValueError(f"Unsupported side: {spawn_side}")

    def update(self, now: float, delta_time: float) -> None:
        if self.state == "queued":
            if now < self.activation_time:
                return
            self.state = "warning"
            self.state_started_at = now
            return

        if self.state == "warning":
            if now - self.state_started_at >= self.warning_duration:
                self.state = "strike"
                self.state_started_at = now
            return

        self.x += self.vx * delta_time

    def draw(self, frame, now: float) -> None:
        if self.state == "queued":
            return

        if self.state == "warning":
            x, y, w, h = self.get_warning_rect()
        else:
            hit_rect = self.get_hit_rect()
            if hit_rect is None:
                return
            x, y, w, h = hit_rect

        if self.state == "warning":
            blend_rect(frame, (x, y, w, h), (0, 215, 255), 0.22)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 3)
            remaining = max(0.0, self.warning_duration - (now - self.state_started_at))
            direction_label = "HOOK ->" if self.spawn_side == "left" else "HOOK <-"
            label = f"{direction_label} {remaining:.1f}"
            label_x = max(10, min(x + 8, frame.shape[1] - 180))
            label_color = (0, 215, 255)
        else:
            blend_rect(frame, (x, y, w, h), (0, 120, 255), 0.4)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 3)
            label = "HOOK"
            label_x = max(10, min(x, frame.shape[1] - 90))
            label_color = (0, 120, 255)

        cv2.putText(
            frame,
            label,
            (label_x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            label_color,
            2,
        )

    def get_warning_rect(self) -> Rect:
        return self.warning_rect

    def get_hit_rect(self) -> Optional[Rect]:
        if self.state != "strike":
            return None
        return int(self.x), int(self.y), self.width, self.height

    def is_finished(self, frame_w: int, frame_h: int) -> bool:
        del frame_h
        if self.state in {"queued", "warning"}:
            return False
        return self.x + self.width < 0 or self.x > frame_w


class PadTarget:
    """A timed pad target that grows, becomes live, then expires if not hit."""

    _cached_image = None
    _image_load_attempted = False

    def __init__(
        self,
        center: Point,
        start_radius: int,
        ready_radius: int,
        grow_duration: float,
        ready_window: float,
        spawned_at: float,
    ):
        self.center = center
        self.start_radius = start_radius
        self.ready_radius = ready_radius
        self.grow_duration = grow_duration
        self.ready_window = ready_window
        self.spawned_at = spawned_at

    def get_current_radius(self, now: float) -> int:
        elapsed = max(0.0, now - self.spawned_at)
        if elapsed >= self.grow_duration:
            return self.ready_radius
        progress = elapsed / max(self.grow_duration, 0.001)
        radius = self.start_radius + (self.ready_radius - self.start_radius) * progress
        return int(round(radius))

    def is_ready(self, now: float) -> bool:
        return (now - self.spawned_at) >= self.grow_duration

    def is_expired(self, now: float) -> bool:
        return (now - self.spawned_at) >= (self.grow_duration + self.ready_window)

    @classmethod
    def get_pad_image(cls):
        """Cache the optional pad image so we do not decode it every frame."""
        if not cls._image_load_attempted:
            cls._cached_image = _load_pad_target_image()
            cls._image_load_attempted = True
        return cls._cached_image

    def draw_pad_image(self, frame, current_radius: int) -> bool:
        """Draw the pad image clipped to the current circular target area."""
        pad_image = self.get_pad_image()
        if pad_image is None or current_radius <= 0:
            return False

        diameter = max(2, current_radius * 2)
        interpolation = cv2.INTER_AREA if diameter <= max(pad_image.shape[:2]) else cv2.INTER_LINEAR
        resized = cv2.resize(pad_image, (diameter, diameter), interpolation=interpolation)

        x = self.center[0] - current_radius
        y = self.center[1] - current_radius
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + diameter)
        y2 = min(frame_h, y + diameter)
        if x1 >= x2 or y1 >= y2:
            return False

        # Clip the resized image to the frame so partially off-screen pads still render cleanly.
        cropped_image = resized[y1 - y : y2 - y, x1 - x : x2 - x]
        circle_mask = np.zeros((diameter, diameter), dtype=np.uint8)
        cv2.circle(circle_mask, (diameter // 2, diameter // 2), max(1, current_radius - 1), 255, -1)
        cropped_mask = circle_mask[y1 - y : y2 - y, x1 - x : x2 - x].astype(np.float32) / 255.0

        if cropped_image.ndim == 2:
            image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR).astype(np.float32)
            image_alpha = cropped_mask
        elif cropped_image.shape[2] == 4:
            image_rgb = cropped_image[:, :, :3].astype(np.float32)
            image_alpha = (cropped_image[:, :, 3].astype(np.float32) / 255.0) * cropped_mask
        else:
            image_rgb = cropped_image[:, :, :3].astype(np.float32)
            image_alpha = cropped_mask

        roi = frame[y1:y2, x1:x2].astype(np.float32)
        alpha = image_alpha[:, :, None]
        blended = image_rgb * alpha + roi * (1.0 - alpha)
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        return True

    def draw(self, frame, now: float) -> None:
        current_radius = self.get_current_radius(now)
        ready = self.is_ready(now)
        pad_color = (255, 0, 0) if ready else (0, 215, 255)
        cx, cy = self.center

        if self.get_pad_image() is not None:
            blend_circle(frame, self.center, current_radius, pad_color, 0.14 if ready else 0.1)
            if self.draw_pad_image(frame, current_radius):
                cv2.circle(frame, self.center, current_radius, pad_color, 3)
            else:
                blend_circle(frame, self.center, current_radius, pad_color, 0.22 if ready else 0.16)
                cv2.circle(frame, self.center, current_radius, pad_color, 3)
                cv2.circle(frame, self.center, max(5, current_radius // 4), pad_color, -1)
        else:
            blend_circle(frame, self.center, current_radius, pad_color, 0.22 if ready else 0.16)
            cv2.circle(frame, self.center, current_radius, pad_color, 3)
            cv2.circle(frame, self.center, max(5, current_radius // 4), pad_color, -1)

        label_y = max(40, cy - self.ready_radius - 34)
        label_x = max(10, min(cx - 70, frame.shape[1] - 180))
        cv2.putText(
            frame,
            "PAD",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            pad_color,
            2,
        )

        if ready:
            remaining = max(0.0, self.ready_window - (now - self.spawned_at - self.grow_duration))
            cv2.putText(
                frame,
                f"HIT {remaining:.1f}",
                (label_x, label_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                pad_color,
                2,
            )
