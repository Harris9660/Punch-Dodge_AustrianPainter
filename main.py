import os
import math
import random
import sys
import threading
import time
from typing import List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_SITE_PACKAGES = os.path.join(
    PROJECT_ROOT,
    ".venv",
    "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages",
)
if os.path.isdir(LOCAL_SITE_PACKAGES) and LOCAL_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, LOCAL_SITE_PACKAGES)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import cv2
import mediapipe as mp
import numpy as np
try:
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
except Exception:
    mp_vision = None
    BaseOptions = None


Rect = Tuple[int, int, int, int]
Point = Tuple[int, int]
ArmSegment = Tuple[Point, Point]


WINDOW_NAME = "Head Dodge Game"
FULLSCREEN_WINDOW = True
TARGET_FPS = 60
TARGET_FRAME_TIME = 1.0 / TARGET_FPS
SHOW_FPS_COUNTER = True
SHOW_PERFORMANCE_BREAKDOWN = True
SHOW_CAMERA_STATS = True
FPS_SMOOTHING = 0.12
CAMERA_BUFFER_SIZE = 1
CAMERA_USE_MJPG = True
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
FACE_DETECTION_SCALE = 0.6
FACE_DETECTION_EVERY_N_FRAMES = 1
FACE_SEARCH_SCALE = 2.5
POSE_INPUT_SCALE = 0.75
POSE_DETECTION_EVERY_N_FRAMES = 2
AUTO_RESTART_ENABLED = True
AUTO_RESTART_DELAY = 1.0
# "pad_work", "dodge", "both"
GAME_MODE = "dodge"
BOTH_MODE_STAGE_DURATION = 10.0
PAD_SCORE_POINTS = 2
PAD_RESPAWN_DELAY = 0.7
PAD_START_RADIUS = 12
PAD_READY_RADIUS = 42
PAD_GROW_DURATION = 0.45
PAD_READY_WINDOW = 1.0
PAD_HEAD_SIDE_OFFSET_RATIO = 0.1
JAB_ACTIVE_DURATION = 0.01
STRAIGHT_SIDE_HEAD_OFFSET_RATIO = 0.25
HOOK_WIDTH = 280
HOOK_HEIGHT = 40
HOOK_WARNING_DURATION = 0.35
HOOK_WARNING_WIDTH = 100
HOOK_WARNING_HEAD_GAP = 18
COMBO_STEP_DELAY = 0.38
ARM_BLOCK_LENGTH_PADDING = 24
ARM_BLOCK_THICKNESS = 56
ARM_DRAW_THICKNESS = 8
FACE_MEMORY_SECONDS = 0.35
FACE_SMOOTHING = 1.0  # 1.0 means no smoothing; use the latest face box immediately.
ARM_GUARD_TOP_MULTIPLIER = 0.7
ARM_GUARD_BOTTOM_MULTIPLIER = 1.0
ARM_MEMORY_SECONDS = 0.12
ARM_PALM_EXTENSION_MULTIPLIER = 0.45
ARM_PALM_EXTENSION_MIN_PIXELS = 18
ARM_SEARCH_SIDE_MULTIPLIER = 1.8
ARM_SEARCH_TOP_MULTIPLIER = 0.5
ARM_SEARCH_BOTTOM_MULTIPLIER = 1.4
ARM_MIN_CONTOUR_AREA = 700
SKIN_YCRCB_LOWER = np.array([0, 133, 77], dtype=np.uint8)
SKIN_YCRCB_UPPER = np.array([255, 173, 127], dtype=np.uint8)
POSE_MODEL_COMPLEXITY = 0
POSE_MIN_PRESENCE_CONFIDENCE = 0.5
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
POSE_VISIBILITY_THRESHOLD = 0.5
POSE_LANDMARKER_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pose_landmarker_lite.task")
if not os.path.exists(POSE_LANDMARKER_MODEL_PATH):
    POSE_LANDMARKER_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pose_landmarker.task")
MEDIAPIPE_SOLUTIONS_AVAILABLE = hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
MEDIAPIPE_TASKS_AVAILABLE = (
    mp_vision is not None
    and BaseOptions is not None
    and os.path.exists(POSE_LANDMARKER_MODEL_PATH)
)
MEDIAPIPE_AVAILABLE = MEDIAPIPE_SOLUTIONS_AVAILABLE or MEDIAPIPE_TASKS_AVAILABLE
POSE_LANDMARK = None
if MEDIAPIPE_SOLUTIONS_AVAILABLE:
    POSE_LANDMARK = mp.solutions.pose.PoseLandmark
elif MEDIAPIPE_TASKS_AVAILABLE:
    POSE_LANDMARK = mp_vision.PoseLandmark


def normalize_game_mode(game_mode: str) -> str:
    """Validate and normalize the selected game mode."""
    normalized_mode = game_mode.strip().lower()
    if normalized_mode not in {"pad_work", "dodge", "both"}:
        raise ValueError("GAME_MODE must be 'pad_work', 'dodge', or 'both'.")
    return normalized_mode


def mode_has_pad_work(game_mode: str) -> bool:
    """Check whether the selected mode includes pad work."""
    return game_mode in {"pad_work", "both"}


def mode_has_dodge(game_mode: str) -> bool:
    """Check whether the selected mode includes dodge attacks."""
    return game_mode in {"dodge", "both"}


def get_mode_label(game_mode: str) -> str:
    """Create a readable label for the selected mode."""
    return {
        "pad_work": "Pad Work",
        "dodge": "Dodge",
        "both": "Both",
    }[game_mode]


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


def create_jab_attack(
    head_rect: Rect,
    frame_w: int,
    frame_h: int,
    start_time: float,
    side: str = "center",
    label: str = "JAB",
) -> JabAttack:
    """Create a telegraphed straight punch aimed at the player's face."""
    head_x, head_y, head_w, head_h = head_rect
    side_offset = int(head_w * STRAIGHT_SIDE_HEAD_OFFSET_RATIO)
    if side == "center":
        center_x = head_x + head_w // 2
    elif side == "left":
        center_x = head_x + side_offset
    elif side == "right":
        center_x = head_x + head_w - side_offset
    else:
        raise ValueError(f"Unsupported straight punch side: {side}")
    center_y = head_y + head_h // 2
    strike_radius = min(PAD_READY_RADIUS, frame_w // 2, frame_h // 2)
    center_x = max(strike_radius, min(frame_w - strike_radius, center_x))
    center_y = max(strike_radius, min(frame_h - strike_radius, center_y))

    return JabAttack(
        center=(center_x, center_y),
        start_radius=PAD_START_RADIUS,
        strike_radius=strike_radius,
        grow_duration=PAD_GROW_DURATION,
        active_duration=JAB_ACTIVE_DURATION,
        start_time=start_time,
        label=label,
    )


def create_hook_attack(
    head_rect: Rect,
    frame_w: int,
    frame_h: int,
    start_time: float,
    spawn_side: Optional[str] = None,
) -> HookAttack:
    """Create a telegraphed horizontal hook aimed at the player's current head level."""
    head_x, head_y, head_w, head_h = head_rect
    width = min(frame_w, HOOK_WIDTH)
    height = min(frame_h, HOOK_HEIGHT)
    warning_width = min(frame_w, HOOK_WARNING_WIDTH)
    head_center_y = head_y + head_h // 2
    y = max(0, min(frame_h - height, head_center_y - height // 2))
    speed = random.uniform(max(frame_w, frame_h) * 1.8, max(frame_w, frame_h) * 2.1)
    spawn_side = spawn_side or random.choice(["left", "right"])

    if spawn_side == "left":
        warning_x = head_x - warning_width - HOOK_WARNING_HEAD_GAP
    else:
        warning_x = head_x + head_w + HOOK_WARNING_HEAD_GAP
    warning_x = max(0, min(frame_w - warning_width, warning_x))
    warning_rect = (warning_x, y, warning_width, height)

    return HookAttack(
        spawn_side=spawn_side,
        speed=speed,
        width=width,
        height=height,
        frame_w=frame_w,
        y=y,
        warning_rect=warning_rect,
        warning_duration=HOOK_WARNING_DURATION,
        start_time=start_time,
    )


def create_one_two_combo(head_rect: Rect, frame_w: int, frame_h: int, start_time: float) -> List[Attack]:
    """Create a 1-2 combo: jab on the right side of the head, then cross on the left."""
    return [
        create_jab_attack(
            head_rect,
            frame_w,
            frame_h,
            start_time=start_time,
            side="right",
            label="JAB",
        ),
        create_jab_attack(
            head_rect,
            frame_w,
            frame_h,
            start_time=start_time + COMBO_STEP_DELAY,
            side="left",
            label="CROSS",
        ),
    ]


def create_one_two_three_combo(head_rect: Rect, frame_w: int, frame_h: int, start_time: float) -> List[Attack]:
    """Create a 1-2-3 combo: jab, cross, then a left hook."""
    return [
        *create_one_two_combo(head_rect, frame_w, frame_h, start_time),
        create_hook_attack(
            head_rect,
            frame_w,
            frame_h,
            start_time=start_time + COMBO_STEP_DELAY * 2,
            spawn_side="right",
        ),
    ]


def create_attack_pattern(head_rect: Rect, frame_w: int, frame_h: int, start_time: float) -> List[Attack]:
    """Create either a single punch or a short combo sequence."""
    pattern_roll = random.random()
    if pattern_roll < 0.35:
        return [create_jab_attack(head_rect, frame_w, frame_h, start_time)]
    if pattern_roll < 0.55:
        return [create_hook_attack(head_rect, frame_w, frame_h, start_time)]
    if pattern_roll < 0.8:
        return create_one_two_combo(head_rect, frame_w, frame_h, start_time)
    return create_one_two_three_combo(head_rect, frame_w, frame_h, start_time)


def intersects(rect1: Rect, rect2: Rect) -> bool:
    """Check if two rectangles overlap."""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (
        (x1 + w1 < x2)
        or (x1 > x2 + w2)
        or (y1 + h1 < y2)
        or (y1 > y2 + h2)
    )


def point_in_rect(x: int, y: int, rect: Rect) -> bool:
    """Check whether a point is inside a rectangle."""
    rect_x, rect_y, rect_w, rect_h = rect
    return rect_x <= x <= rect_x + rect_w and rect_y <= y <= rect_y + rect_h


def point_distance(point1: Point, point2: Point) -> float:
    """Measure the 2D distance between two points."""
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])


def segment_hits_circle(start: Point, end: Point, center: Point, radius: float) -> bool:
    """Check whether a line segment intersects a circle."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return point_distance(start, center) <= radius

    t = ((center[0] - start[0]) * dx + (center[1] - start[1]) * dy) / length_squared
    t = max(0.0, min(1.0, t))
    closest_point = (
        start[0] + dx * t,
        start[1] + dy * t,
    )
    return math.hypot(closest_point[0] - center[0], closest_point[1] - center[1]) <= radius


def rect_center(rect: Rect) -> Point:
    """Return the center point of a rectangle."""
    x, y, w, h = rect
    return x + w // 2, y + h // 2


def smooth_rect(previous_rect: Rect, current_rect: Rect, smoothing: float) -> Rect:
    """Blend face boxes so the tracked head rectangle feels stable."""
    smoothing = max(0.0, min(1.0, smoothing))
    inverse = 1.0 - smoothing
    return (
        int(round(previous_rect[0] * inverse + current_rect[0] * smoothing)),
        int(round(previous_rect[1] * inverse + current_rect[1] * smoothing)),
        int(round(previous_rect[2] * inverse + current_rect[2] * smoothing)),
        int(round(previous_rect[3] * inverse + current_rect[3] * smoothing)),
    )


def clamp_rect(x: int, y: int, width: int, height: int, frame_w: int, frame_h: int) -> Rect:
    """Clamp a rectangle to remain inside the frame."""
    x = max(0, min(frame_w - 1, x))
    y = max(0, min(frame_h - 1, y))
    width = max(1, min(width, frame_w - x))
    height = max(1, min(height, frame_h - y))
    return x, y, width, height


def blend_rect(frame, rect: Rect, color: Tuple[int, int, int], alpha: float) -> None:
    """Blend a rectangle into only the affected pixels."""
    frame_h, frame_w = frame.shape[:2]
    x, y, width, height = clamp_rect(*rect, frame_w, frame_h)
    roi = frame[y : y + height, x : x + width]
    overlay = roi.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def blend_circle(frame, center: Point, radius: int, color: Tuple[int, int, int], alpha: float) -> None:
    """Blend a circle into only the affected pixels."""
    if radius <= 0:
        return

    frame_h, frame_w = frame.shape[:2]
    x, y, width, height = clamp_rect(
        center[0] - radius,
        center[1] - radius,
        radius * 2,
        radius * 2,
        frame_w,
        frame_h,
    )
    roi = frame[y : y + height, x : x + width]
    overlay = roi.copy()
    cv2.circle(overlay, (center[0] - x, center[1] - y), radius, color, -1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def smooth_metric(previous_value: float, current_value: float, smoothing: float) -> float:
    """Smooth noisy on-screen timing metrics."""
    if previous_value <= 0.0:
        return current_value
    return previous_value * (1.0 - smoothing) + current_value * smoothing


class LatestFrameCapture:
    """Continuously read camera frames so the game loop does not block on capture."""

    def __init__(self, capture, initial_frame) -> None:
        self.capture = capture
        self.latest_frame = initial_frame
        self.latest_frame_id = 0
        self.latest_read_ms = 0.0
        self.live_fps = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_frame_at = time.perf_counter()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def read_latest(self):
        """Return the newest captured frame plus camera timing information."""
        with self._lock:
            return self.latest_frame, self.latest_frame_id, self.latest_read_ms, self.live_fps

    def _reader_loop(self) -> None:
        while self._running:
            read_started_at = time.perf_counter()
            ret, frame = self.capture.read()
            read_finished_at = time.perf_counter()
            if not ret:
                time.sleep(0.001)
                continue

            read_elapsed_ms = (read_finished_at - read_started_at) * 1000.0
            current_live_fps = 0.0
            frame_interval = read_finished_at - self._last_frame_at
            if frame_interval > 0.0:
                current_live_fps = 1.0 / frame_interval
            self._last_frame_at = read_finished_at

            with self._lock:
                self.latest_frame = frame
                self.latest_frame_id += 1
                self.latest_read_ms = read_elapsed_ms
                if current_live_fps > 0.0:
                    self.live_fps = smooth_metric(self.live_fps, current_live_fps, 0.15)


def get_face_search_rect(head_rect: Rect, frame_w: int, frame_h: int) -> Rect:
    """Expand the previous face box into a search window for the next detection."""
    center_x, center_y = rect_center(head_rect)
    search_width = max(head_rect[2], int(round(head_rect[2] * FACE_SEARCH_SCALE)))
    search_height = max(head_rect[3], int(round(head_rect[3] * FACE_SEARCH_SCALE)))
    return clamp_rect(
        center_x - search_width // 2,
        center_y - search_height // 2,
        search_width,
        search_height,
        frame_w,
        frame_h,
    )


def get_arm_search_rects(head_rect: Rect, frame_w: int, frame_h: int) -> List[Rect]:
    """Build left and right guard search zones around the head."""
    head_x, head_y, head_w, head_h = head_rect
    side_width = int(head_w * ARM_SEARCH_SIDE_MULTIPLIER)
    top = max(0, head_y - int(head_h * ARM_SEARCH_TOP_MULTIPLIER))
    bottom = min(frame_h, head_y + head_h + int(head_h * ARM_SEARCH_BOTTOM_MULTIPLIER))
    search_height = max(1, bottom - top)

    left_rect = clamp_rect(
        head_x - side_width,
        top,
        side_width + int(head_w * 0.2),
        search_height,
        frame_w,
        frame_h,
    )
    right_rect = clamp_rect(
        head_x + int(head_w * 0.8),
        top,
        side_width + int(head_w * 0.2),
        search_height,
        frame_w,
        frame_h,
    )
    return [left_rect, right_rect]


def landmark_is_visible(landmark) -> bool:
    """Check whether a MediaPipe landmark is usable for blocking."""
    return (
        0.0 <= landmark.x <= 1.0
        and 0.0 <= landmark.y <= 1.0
        and getattr(landmark, "visibility", 0.0) >= POSE_VISIBILITY_THRESHOLD
    )


def landmark_to_point(landmark, frame_w: int, frame_h: int) -> Point:
    """Convert a normalized MediaPipe landmark to a frame pixel coordinate."""
    x = int(min(max(landmark.x, 0.0), 1.0) * (frame_w - 1))
    y = int(min(max(landmark.y, 0.0), 1.0) * (frame_h - 1))
    return x, y


def extend_to_palm(elbow_point: Point, wrist_point: Point, frame_w: int, frame_h: int) -> Point:
    """Extend the forearm line beyond the wrist so blocking reaches toward the palm."""
    dx = wrist_point[0] - elbow_point[0]
    dy = wrist_point[1] - elbow_point[1]
    length = math.hypot(dx, dy)
    if length < 1.0:
        return wrist_point

    extension = max(ARM_PALM_EXTENSION_MIN_PIXELS, int(length * ARM_PALM_EXTENSION_MULTIPLIER))
    extended_x = int(round(wrist_point[0] + (dx / length) * extension))
    extended_y = int(round(wrist_point[1] + (dy / length) * extension))
    extended_x = max(0, min(frame_w - 1, extended_x))
    extended_y = max(0, min(frame_h - 1, extended_y))
    return extended_x, extended_y


def get_pose_landmarks(pose_results):
    """Extract landmarks from either MediaPipe Solutions or MediaPipe Tasks results."""
    if pose_results is None:
        return None

    pose_landmarks = getattr(pose_results, "pose_landmarks", None)
    if pose_landmarks is None:
        return None

    if hasattr(pose_landmarks, "landmark"):
        return pose_landmarks.landmark
    if isinstance(pose_landmarks, list) and pose_landmarks:
        return pose_landmarks[0]
    return None


def expand_block_rect(
    x: int,
    y: int,
    width: int,
    height: int,
    frame_w: int,
    frame_h: int,
) -> Rect:
    """Inflate a blocking area into a wider, more rectangular guard zone."""
    x -= ARM_BLOCK_LENGTH_PADDING
    y -= ARM_BLOCK_LENGTH_PADDING
    width += ARM_BLOCK_LENGTH_PADDING * 2
    height += ARM_BLOCK_LENGTH_PADDING * 2

    center_x = x + width / 2.0
    center_y = y + height / 2.0
    width = max(width, ARM_BLOCK_THICKNESS)
    height = max(height, ARM_BLOCK_THICKNESS)
    x = int(round(center_x - width / 2.0))
    y = int(round(center_y - height / 2.0))
    return clamp_rect(x, y, int(width), int(height), frame_w, frame_h)


def segment_to_rect(start: Point, end: Point, frame_w: int, frame_h: int) -> Rect:
    """Expand a line segment into a wider rectangular blocking area."""
    x = min(start[0], end[0])
    y = min(start[1], end[1])
    width = abs(start[0] - end[0])
    height = abs(start[1] - end[1])
    return expand_block_rect(x, y, width, height, frame_w, frame_h)


def detect_arm_blocks(pose_results, head_rect: Rect, frame_w: int, frame_h: int) -> Tuple[List[Rect], List[ArmSegment]]:
    """Build hook-blocking arm segments from MediaPipe pose landmarks."""
    landmarks = get_pose_landmarks(pose_results)
    if landmarks is None or POSE_LANDMARK is None:
        return [], []

    head_y = head_rect[1]
    head_h = head_rect[3]
    guard_top = head_y - int(head_h * ARM_GUARD_TOP_MULTIPLIER)
    guard_bottom = head_y + head_h + int(head_h * ARM_GUARD_BOTTOM_MULTIPLIER)
    arm_rects: List[Rect] = []
    arm_segments: List[ArmSegment] = []

    for elbow_index, wrist_index in (
        (POSE_LANDMARK.LEFT_ELBOW.value, POSE_LANDMARK.LEFT_WRIST.value),
        (POSE_LANDMARK.RIGHT_ELBOW.value, POSE_LANDMARK.RIGHT_WRIST.value),
    ):
        elbow = landmarks[elbow_index]
        wrist = landmarks[wrist_index]
        if not landmark_is_visible(elbow) or not landmark_is_visible(wrist):
            continue

        elbow_point = landmark_to_point(elbow, frame_w, frame_h)
        wrist_point = landmark_to_point(wrist, frame_w, frame_h)
        palm_point = extend_to_palm(elbow_point, wrist_point, frame_w, frame_h)
        segment_top = min(elbow_point[1], palm_point[1])
        segment_bottom = max(elbow_point[1], palm_point[1])
        if segment_bottom < guard_top or segment_top > guard_bottom:
            continue

        arm_segments.append((elbow_point, palm_point))
        arm_rects.append(segment_to_rect(elbow_point, palm_point, frame_w, frame_h))

    return arm_rects, arm_segments


def detect_arm_blocks_fallback(
    frame,
    gray,
    head_rect: Rect,
    previous_gray: Optional[np.ndarray],
    frame_w: int,
    frame_h: int,
) -> Tuple[List[Rect], List[ArmSegment]]:
    """Fallback arm detection using motion and skin cues near the head."""
    blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, SKIN_YCRCB_LOWER, SKIN_YCRCB_UPPER)

    motion_mask = np.zeros_like(gray)
    if previous_gray is not None and previous_gray.shape == blurred_gray.shape:
        diff = cv2.absdiff(blurred_gray, previous_gray)
        _, motion_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    arm_rects: List[Rect] = []
    arm_segments: List[ArmSegment] = []
    kernel = np.ones((5, 5), dtype=np.uint8)

    for search_rect in get_arm_search_rects(head_rect, frame_w, frame_h):
        x, y, w, h = search_rect
        roi_skin = skin_mask[y : y + h, x : x + w]
        roi_motion = motion_mask[y : y + h, x : x + w]
        combined_mask = cv2.bitwise_or(roi_skin, roi_motion)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = [contour for contour in contours if cv2.contourArea(contour) >= ARM_MIN_CONTOUR_AREA]
        if not candidates:
            continue

        contour = max(candidates, key=cv2.contourArea)
        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
        padded_rect = expand_block_rect(
            x + rect_x,
            y + rect_y,
            rect_w,
            rect_h,
            frame_w,
            frame_h,
        )
        arm_rects.append(padded_rect)

        mid_y = padded_rect[1] + padded_rect[3] // 2
        arm_segments.append(((padded_rect[0], mid_y), (padded_rect[0] + padded_rect[2], mid_y)))

    return arm_rects, arm_segments


def create_pad_target(head_rect: Rect, frame_w: int, frame_h: int, spawned_at: float) -> PadTarget:
    """Create a timed pad-work target near head height at a reachable striking distance."""
    head_center_x, head_center_y = rect_center(head_rect)
    head_w = head_rect[2]
    side_offset = int(head_w * PAD_HEAD_SIDE_OFFSET_RATIO)
    side = random.choice(["left", "right"])
    center_x = head_rect[0] - side_offset if side == "left" else head_rect[0] + head_w + side_offset
    center_y = head_center_y

    center_x = max(PAD_READY_RADIUS + 10, min(frame_w - PAD_READY_RADIUS - 10, center_x))
    center_y = max(PAD_READY_RADIUS + 40, min(frame_h - PAD_READY_RADIUS - 10, center_y))
    return PadTarget(
        center=(center_x, center_y),
        start_radius=PAD_START_RADIUS,
        ready_radius=PAD_READY_RADIUS,
        grow_duration=PAD_GROW_DURATION,
        ready_window=PAD_READY_WINDOW,
        spawned_at=spawned_at,
    )


def main() -> None:
    """Run the head-dodging game."""
    game_mode = normalize_game_mode(GAME_MODE)
    current_stage = "dodge" if game_mode == "both" else game_mode
    stage_started_at = time.time()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access webcam.")
        return
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    if CAMERA_USE_MJPG and hasattr(cv2, "VideoWriter_fourcc"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from webcam.")
        return
    frame_h, frame_w = frame.shape[:2]
    capture_stream = LatestFrameCapture(cap, frame)
    capture_stream.start()
    camera_reported_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_reported_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_reported_fps = cap.get(cv2.CAP_PROP_FPS)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    pose_tracker = None
    pose_tracker_backend = None
    pose_status_text = None
    if MEDIAPIPE_SOLUTIONS_AVAILABLE:
        pose_tracker = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=POSE_MODEL_COMPLEXITY,
            smooth_landmarks=True,
            min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
        )
        pose_tracker_backend = "solutions"
    elif MEDIAPIPE_TASKS_AVAILABLE:
        try:
            pose_tracker = mp_vision.PoseLandmarker.create_from_options(
                mp_vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=POSE_LANDMARKER_MODEL_PATH,
                        delegate=BaseOptions.Delegate.CPU,
                    ),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
                    min_pose_presence_confidence=POSE_MIN_PRESENCE_CONFIDENCE,
                    min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
                )
            )
            pose_tracker_backend = "tasks"
        except Exception as error:
            pose_status_text = "MediaPipe pose unavailable: using fallback arm blocks"
            print(f"{pose_status_text}. {error}")
    else:
        pose_status_text = "MediaPipe pose unavailable: using fallback arm blocks"

    mouse_state = {
        "restart_requested": False,
        "restart_button_rect": None,
    }

    def on_mouse(event, x, y, flags, param) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        button_rect = mouse_state["restart_button_rect"]
        if button_rect is not None and point_in_rect(x, y, button_rect):
            mouse_state["restart_requested"] = True

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if FULLSCREEN_WINDOW:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    attacks: List[Attack] = []
    pad_target: Optional[PadTarget] = None
    next_pad_spawn_time = time.time()
    spawn_interval = 1.5
    last_spawn_time = time.time()
    score = 0
    game_over = False
    game_over_started_at: Optional[float] = None
    last_head_rect: Optional[Rect] = None
    last_head_seen_at = 0.0
    last_arm_rects: List[Rect] = []
    last_arm_segments: List[ArmSegment] = []
    last_arm_seen_at = 0.0
    last_pose_results = None
    previous_gray: Optional[np.ndarray] = None
    previous_frame_time = time.perf_counter() - TARGET_FRAME_TIME
    last_capture_frame_id = -1
    fps_display = float(TARGET_FPS)
    frame_ms_display = 0.0
    read_ms_display = 0.0
    pose_ms_display = 0.0
    face_ms_display = 0.0
    live_camera_fps_display = 0.0

    def clear_stage_elements(started_at: float) -> None:
        nonlocal attacks, pad_target, next_pad_spawn_time, last_spawn_time

        attacks = []
        pad_target = None
        next_pad_spawn_time = started_at
        last_spawn_time = started_at

    def reset_round(current_head_rect: Optional[Rect]) -> None:
        nonlocal score, game_over, game_over_started_at, current_stage, stage_started_at
        nonlocal last_head_rect, last_head_seen_at, last_arm_rects, last_arm_segments, last_arm_seen_at
        nonlocal last_pose_results, previous_gray

        current_stage = "dodge" if game_mode == "both" else game_mode
        stage_started_at = time.time()
        clear_stage_elements(stage_started_at)
        score = 0
        game_over = False
        game_over_started_at = None
        last_head_rect = current_head_rect
        last_head_seen_at = time.time() if current_head_rect is not None else 0.0
        last_arm_rects = []
        last_arm_segments = []
        last_arm_seen_at = 0.0
        last_pose_results = None
        previous_gray = None
        mouse_state["restart_requested"] = False

    while True:
        frame_started_at = time.perf_counter()
        delta_time = frame_started_at - previous_frame_time
        previous_frame_time = frame_started_at
        fps_display = smooth_metric(
            fps_display,
            (1.0 / delta_time) if delta_time > 0.0 else 0.0,
            FPS_SMOOTHING,
        )

        frame, capture_frame_id, capture_read_ms, live_camera_fps = capture_stream.read_latest()
        if frame is None:
            break
        has_new_camera_frame = capture_frame_id != last_capture_frame_id
        if has_new_camera_frame:
            last_capture_frame_id = capture_frame_id

        now = time.time()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        read_elapsed = capture_read_ms / 1000.0
        if live_camera_fps > 0.0:
            live_camera_fps_display = smooth_metric(live_camera_fps_display, live_camera_fps, FPS_SMOOTHING)

        if game_mode == "both" and not game_over and (now - stage_started_at) >= BOTH_MODE_STAGE_DURATION:
            current_stage = "pad_work" if current_stage == "dodge" else "dodge"
            stage_started_at = now
            clear_stage_elements(now)

        pad_work_enabled = mode_has_pad_work(current_stage)
        dodge_enabled = mode_has_dodge(current_stage)

        face_elapsed = 0.0
        largest_face = None
        if has_new_camera_frame and (capture_frame_id % max(1, FACE_DETECTION_EVERY_N_FRAMES)) == 0:
            face_input = gray
            if FACE_DETECTION_SCALE != 1.0:
                face_input = cv2.resize(
                    gray,
                    None,
                    fx=FACE_DETECTION_SCALE,
                    fy=FACE_DETECTION_SCALE,
                    interpolation=cv2.INTER_LINEAR,
                )

            search_rect = None
            if last_head_rect is not None:
                search_rect = get_face_search_rect(
                    (
                        int(round(last_head_rect[0] * FACE_DETECTION_SCALE)),
                        int(round(last_head_rect[1] * FACE_DETECTION_SCALE)),
                        max(1, int(round(last_head_rect[2] * FACE_DETECTION_SCALE))),
                        max(1, int(round(last_head_rect[3] * FACE_DETECTION_SCALE))),
                    ) if FACE_DETECTION_SCALE != 1.0 else last_head_rect,
                    face_input.shape[1],
                    face_input.shape[0],
                )

            face_started_at = time.perf_counter()
            if search_rect is not None:
                search_x, search_y, search_w, search_h = search_rect
                search_roi = face_input[search_y : search_y + search_h, search_x : search_x + search_w]
                local_faces = face_cascade.detectMultiScale(search_roi, scaleFactor=1.3, minNeighbors=5)
                if len(local_faces) > 0:
                    local_largest_face = max(local_faces, key=lambda face: face[2] * face[3])
                    largest_face = (
                        int(local_largest_face[0] + search_x),
                        int(local_largest_face[1] + search_y),
                        int(local_largest_face[2]),
                        int(local_largest_face[3]),
                    )

            if largest_face is None:
                faces = face_cascade.detectMultiScale(face_input, scaleFactor=1.3, minNeighbors=5)
                if len(faces) > 0:
                    largest_face = max(faces, key=lambda face: face[2] * face[3])
            face_elapsed = time.perf_counter() - face_started_at

        head_rect: Optional[Rect] = None
        if largest_face is not None:
            if FACE_DETECTION_SCALE != 1.0:
                scale = 1.0 / FACE_DETECTION_SCALE
                head_rect = (
                    int(round(largest_face[0] * scale)),
                    int(round(largest_face[1] * scale)),
                    int(round(largest_face[2] * scale)),
                    int(round(largest_face[3] * scale)),
                )
            else:
                head_rect = tuple(int(value) for value in largest_face)
            head_rect = clamp_rect(*head_rect, frame_w, frame_h)
            if last_head_rect is None:
                last_head_rect = head_rect
            else:
                last_head_rect = smooth_rect(last_head_rect, head_rect, FACE_SMOOTHING)
            last_head_seen_at = now
        elif last_head_rect is not None and (now - last_head_seen_at) > FACE_MEMORY_SECONDS:
            last_head_rect = None

        target_head_rect = last_head_rect
        pose_results = last_pose_results
        pose_elapsed = 0.0
        if (
            pose_tracker is not None
            and target_head_rect is not None
            and has_new_camera_frame
            and (
                last_pose_results is None
                or (capture_frame_id % max(1, POSE_DETECTION_EVERY_N_FRAMES)) == 0
            )
        ):
            pose_input_frame = frame
            if POSE_INPUT_SCALE != 1.0:
                pose_input_frame = cv2.resize(
                    frame,
                    None,
                    fx=POSE_INPUT_SCALE,
                    fy=POSE_INPUT_SCALE,
                    interpolation=cv2.INTER_LINEAR,
                )
            pose_rgb_frame = cv2.cvtColor(pose_input_frame, cv2.COLOR_BGR2RGB)
            pose_started_at = time.perf_counter()
            if pose_tracker_backend == "solutions":
                pose_results = pose_tracker.process(pose_rgb_frame)
            elif pose_tracker_backend == "tasks":
                frame_timestamp_ms = int(frame_started_at * 1000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=pose_rgb_frame)
                pose_results = pose_tracker.detect_for_video(mp_image, frame_timestamp_ms)
            pose_elapsed = time.perf_counter() - pose_started_at
            last_pose_results = pose_results
        elif target_head_rect is None:
            pose_results = None
            last_pose_results = None

        arm_rects: List[Rect] = []
        arm_segments: List[ArmSegment] = []
        if target_head_rect is not None:
            if pose_tracker is not None:
                arm_rects, arm_segments = detect_arm_blocks(pose_results, target_head_rect, frame_w, frame_h)
            else:
                arm_rects, arm_segments = detect_arm_blocks_fallback(
                    frame,
                    gray,
                    target_head_rect,
                    previous_gray,
                    frame_w,
                    frame_h,
                )
            if arm_rects:
                last_arm_rects = arm_rects
                last_arm_segments = arm_segments
                last_arm_seen_at = now
            elif now - last_arm_seen_at <= ARM_MEMORY_SECONDS:
                arm_rects = last_arm_rects
                arm_segments = last_arm_segments
            else:
                last_arm_rects = []
                last_arm_segments = []

        if (
            pad_work_enabled
            and not game_over
            and target_head_rect is not None
            and pad_target is None
            and now >= next_pad_spawn_time
        ):
            pad_target = create_pad_target(target_head_rect, frame_w, frame_h, now)

        if dodge_enabled and not game_over and target_head_rect is not None and (now - last_spawn_time > spawn_interval):
            attacks.extend(create_attack_pattern(target_head_rect, frame_w, frame_h, now))
            last_spawn_time = now

        if dodge_enabled and not game_over:
            remaining_attacks: List[Attack] = []
            for attack in attacks:
                attack.update(now, delta_time)
                hit_rect = attack.get_hit_rect()

                if isinstance(attack, (JabAttack, HookAttack)) and hit_rect is not None:
                    if any(intersects(arm_rect, hit_rect) for arm_rect in arm_rects):
                        continue

                if target_head_rect is not None and hit_rect is not None and intersects(target_head_rect, hit_rect):
                    game_over = True
                    if game_over_started_at is None:
                        game_over_started_at = now

                if attack.is_finished(frame_w, frame_h):
                    score += 1
                else:
                    remaining_attacks.append(attack)
            attacks = remaining_attacks

        if pad_work_enabled and not game_over and pad_target is not None:
            if (
                pad_target.is_ready(now)
                and any(
                    segment_hits_circle(arm_start, arm_end, pad_target.center, pad_target.ready_radius)
                    for arm_start, arm_end in arm_segments
                )
            ):
                score += PAD_SCORE_POINTS
                pad_target = None
                next_pad_spawn_time = now + PAD_RESPAWN_DELAY
            elif pad_target.is_expired(now):
                pad_target = None
                next_pad_spawn_time = now + PAD_RESPAWN_DELAY

        if target_head_rect is not None:
            x, y, w, h = target_head_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for arm_start, arm_end in arm_segments:
            cv2.line(frame, arm_start, arm_end, (255, 255, 0), ARM_DRAW_THICKNESS)
            cv2.circle(frame, arm_start, ARM_DRAW_THICKNESS + 2, (255, 255, 0), -1)
            cv2.circle(frame, arm_end, ARM_DRAW_THICKNESS + 2, (255, 255, 0), -1)

        for attack in attacks:
            attack.draw(frame, now)

        if pad_work_enabled and not game_over and pad_target is not None:
            pad_target.draw(frame, now)

        cv2.putText(
            frame,
            f"Score: {score}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Mode: {get_mode_label(game_mode)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        if game_mode == "both":
            stage_remaining = max(0.0, BOTH_MODE_STAGE_DURATION - (now - stage_started_at))
            cv2.putText(
                frame,
                f"Stage: {get_mode_label(current_stage)} {stage_remaining:.1f}s",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            if current_stage == "pad_work":
                legend_text = "Yellow pad grows, then touch the blue pad with a cyan arm line"
            else:
                legend_text = "Yellow = warnings | Cyan guard blocks jabs/hooks | Orange = live hook"
            legend_y = 120
        elif game_mode == "pad_work":
            legend_text = "Yellow pad grows, then touch the blue pad with a cyan arm line"
            legend_y = 90
        else:
            legend_text = "Yellow = warnings | Cyan guard blocks jabs/hooks | Orange = live hook"
            legend_y = 90
        cv2.putText(
            frame,
            legend_text,
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        if pose_tracker is None:
            cv2.putText(
                frame,
                pose_status_text or "MediaPipe pose unavailable: hook blocking off",
                (10, legend_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        if SHOW_FPS_COUNTER:
            cv2.putText(
                frame,
                f"FPS: {fps_display:.1f}",
                (max(10, frame_w - 170), 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        if SHOW_CAMERA_STATS:
            cv2.putText(
                frame,
                f"Cam {camera_reported_width}x{camera_reported_height} @ {camera_reported_fps:.1f} live {live_camera_fps_display:.1f}",
                (max(10, frame_w - 320), 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                2,
            )

        if dodge_enabled and game_over:
            button_width = int(frame_w * 0.24)
            button_height = int(frame_h * 0.1)
            button_x = int((frame_w - button_width) / 2)
            button_y = int(frame_h * 0.68)
            restart_button_rect = (button_x, button_y, button_width, button_height)
            mouse_state["restart_button_rect"] = restart_button_rect

            cv2.putText(
                frame,
                f"Game Over! Final Score: {score}",
                (int(frame_w * 0.08), int(frame_h * 0.5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )
            cv2.putText(
                frame,
                "Click Restart or press q to quit",
                (int(frame_w * 0.09), int(frame_h * 0.6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
            if AUTO_RESTART_ENABLED and game_over_started_at is not None:
                restart_in = max(0.0, AUTO_RESTART_DELAY - (now - game_over_started_at))
                cv2.putText(
                    frame,
                    f"Auto restart in {restart_in:.1f}s",
                    (int(frame_w * 0.18), int(frame_h * 0.64)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
            cv2.rectangle(
                frame,
                (button_x, button_y),
                (button_x + button_width, button_y + button_height),
                (40, 180, 40),
                -1,
            )
            cv2.rectangle(
                frame,
                (button_x, button_y),
                (button_x + button_width, button_y + button_height),
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Restart",
                (button_x + int(button_width * 0.19), button_y + int(button_height * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
        else:
            mouse_state["restart_button_rect"] = None

        frame_elapsed = time.perf_counter() - frame_started_at
        frame_ms_display = smooth_metric(frame_ms_display, frame_elapsed * 1000.0, FPS_SMOOTHING)
        read_ms_display = smooth_metric(read_ms_display, read_elapsed * 1000.0, FPS_SMOOTHING)
        pose_ms_display = smooth_metric(pose_ms_display, pose_elapsed * 1000.0, FPS_SMOOTHING)
        face_ms_display = smooth_metric(face_ms_display, face_elapsed * 1000.0, FPS_SMOOTHING)
        if SHOW_PERFORMANCE_BREAKDOWN:
            cv2.putText(
                frame,
                f"Frame {frame_ms_display:.1f}ms | Read {read_ms_display:.1f} | Pose {pose_ms_display:.1f} | Face {face_ms_display:.1f}",
                (10, frame_h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        cv2.imshow(WINDOW_NAME, frame)

        remaining_frame_time = TARGET_FRAME_TIME - frame_elapsed
        if remaining_frame_time > 0:
            time.sleep(remaining_frame_time)

        key = cv2.waitKey(1) & 0xFF
        if (
            dodge_enabled
            and
            AUTO_RESTART_ENABLED
            and game_over
            and game_over_started_at is not None
            and (now - game_over_started_at) >= AUTO_RESTART_DELAY
        ):
            mouse_state["restart_requested"] = True
        if mouse_state["restart_requested"]:
            reset_round(target_head_rect)
        if key == ord("q"):
            break

        previous_gray = cv2.GaussianBlur(gray, (7, 7), 0)

    capture_stream.stop()
    cap.release()
    if pose_tracker is not None:
        pose_tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
