from typing import Optional

import cv2

from game_utils import Point, Rect, blend_circle, blend_rect


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

    def draw(self, frame, now: float) -> None:
        current_radius = self.get_current_radius(now)
        ready = self.is_ready(now)
        pad_color = (255, 0, 0) if ready else (0, 215, 255)
        cx, cy = self.center

        blend_circle(frame, self.center, current_radius, pad_color, 0.22 if ready else 0.16)
        cv2.circle(frame, self.center, current_radius, pad_color, 3)
        cv2.circle(frame, self.center, max(5, current_radius // 4), pad_color, -1)

        label_y = max(30, cy - self.ready_radius - 12)
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
