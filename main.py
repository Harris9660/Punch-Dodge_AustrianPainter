import os
import math
import random
import sys
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
FULLSCREEN_WINDOW = False
TARGET_FPS = 60
TARGET_FRAME_TIME = 1.0 / TARGET_FPS
JAB_BOX_WIDTH = 100
JAB_BOX_HEIGHT = 100
HOOK_WIDTH = 280
HOOK_HEIGHT = 40
HOOK_WARNING_DURATION = 0.35
HOOK_WARNING_WIDTH = 100
HOOK_WARNING_HEAD_GAP = 18
ARM_BLOCK_PADDING = 20
ARM_DRAW_THICKNESS = 6
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
    """A jab shows a warning box before it becomes dangerous."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        warning_duration: float,
        active_duration: float,
        start_time: float,
    ):
        self.rect = (x, y, width, height)
        self.warning_duration = warning_duration
        self.active_duration = active_duration
        self.state = "warning"
        self.state_started_at = start_time
        self.finished = False

    def update(self, now: float, delta_time: float) -> None:
        del delta_time
        if self.finished:
            return

        elapsed = now - self.state_started_at
        if self.state == "warning" and elapsed >= self.warning_duration:
            self.state = "strike"
            self.state_started_at = now
        elif self.state == "strike" and elapsed >= self.active_duration:
            self.finished = True

    def draw(self, frame, now: float) -> None:
        x, y, w, h = self.rect
        overlay = frame.copy()

        if self.state == "warning":
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 215, 255), -1)
            cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 215, 255), 3)

            remaining = max(0.0, self.warning_duration - (now - self.state_started_at))
            cv2.putText(
                frame,
                f"JAB IN {remaining:.1f}",
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 215, 255),
                2,
            )
        else:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(
                frame,
                "MOVE!",
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

    def get_hit_rect(self) -> Optional[Rect]:
        if self.state == "strike" and not self.finished:
            return self.rect
        return None

    def is_finished(self, frame_w: int, frame_h: int) -> bool:
        del frame_w, frame_h
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
        self.warning_duration = warning_duration
        self.state = "warning"
        self.state_started_at = start_time

        if spawn_side == "left":
            self.x = -width
            self.vx = speed
        elif spawn_side == "right":
            self.x = frame_w
            self.vx = -speed
        else:
            raise ValueError(f"Unsupported side: {spawn_side}")

    def update(self, now: float, delta_time: float) -> None:
        if self.state == "warning":
            if now - self.state_started_at >= self.warning_duration:
                self.state = "strike"
                self.state_started_at = now
            return

        self.x += self.vx * delta_time

    def draw(self, frame, now: float) -> None:
        if self.state == "warning":
            x, y, w, h = self.get_warning_rect()
        else:
            hit_rect = self.get_hit_rect()
            if hit_rect is None:
                return
            x, y, w, h = hit_rect

        overlay = frame.copy()
        if self.state == "warning":
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 215, 255), -1)
            cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 3)
            remaining = max(0.0, self.warning_duration - (now - self.state_started_at))
            direction_label = "HOOK ->" if self.spawn_side == "left" else "HOOK <-"
            label = f"{direction_label} {remaining:.1f}"
            label_x = max(10, min(x + 8, frame.shape[1] - 180))
            label_color = (0, 215, 255)
        else:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 120, 255), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
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
        if self.state != "strike":
            return False
        return self.x + self.width < 0 or self.x > frame_w


def create_jab_attack(head_rect: Rect, frame_w: int, frame_h: int, start_time: float) -> JabAttack:
    """Create a telegraphed jab centered on the player's current head position."""
    head_x, head_y, head_w, head_h = head_rect
    width = min(frame_w, JAB_BOX_WIDTH)
    height = min(frame_h, JAB_BOX_HEIGHT)
    center_x = head_x + head_w // 2
    center_y = head_y + head_h // 2
    x = max(0, min(frame_w - width, center_x - width // 2))
    y = max(0, min(frame_h - height, center_y - height // 2))

    return JabAttack(
        x=x,
        y=y,
        width=width,
        height=height,
        warning_duration=0.75,
        active_duration=0.18,
        start_time=start_time,
    )


def create_hook_attack(head_rect: Rect, frame_w: int, frame_h: int, start_time: float) -> HookAttack:
    """Create a telegraphed horizontal hook aimed at the player's current head level."""
    head_x, head_y, head_w, head_h = head_rect
    width = min(frame_w, HOOK_WIDTH)
    height = min(frame_h, HOOK_HEIGHT)
    warning_width = min(frame_w, HOOK_WARNING_WIDTH)
    head_center_y = head_y + head_h // 2
    y = max(0, min(frame_h - height, head_center_y - height // 2))
    speed = random.uniform(max(frame_w, frame_h) * 1.8, max(frame_w, frame_h) * 2.1)
    spawn_side = random.choice(["left", "right"])

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


def clamp_rect(x: int, y: int, width: int, height: int, frame_w: int, frame_h: int) -> Rect:
    """Clamp a rectangle to remain inside the frame."""
    x = max(0, min(frame_w - 1, x))
    y = max(0, min(frame_h - 1, y))
    width = max(1, min(width, frame_w - x))
    height = max(1, min(height, frame_h - y))
    return x, y, width, height


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


def segment_to_rect(start: Point, end: Point, padding: int, frame_w: int, frame_h: int) -> Rect:
    """Expand a line segment into a blocking rectangle."""
    x = min(start[0], end[0]) - padding
    y = min(start[1], end[1]) - padding
    width = abs(start[0] - end[0]) + padding * 2
    height = abs(start[1] - end[1]) + padding * 2
    return clamp_rect(x, y, width, height, frame_w, frame_h)


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
        arm_rects.append(segment_to_rect(elbow_point, palm_point, ARM_BLOCK_PADDING, frame_w, frame_h))

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
        padded_rect = clamp_rect(
            x + rect_x - ARM_BLOCK_PADDING,
            y + rect_y - ARM_BLOCK_PADDING,
            rect_w + ARM_BLOCK_PADDING * 2,
            rect_h + ARM_BLOCK_PADDING * 2,
            frame_w,
            frame_h,
        )
        arm_rects.append(padded_rect)

        mid_y = padded_rect[1] + padded_rect[3] // 2
        arm_segments.append(((padded_rect[0], mid_y), (padded_rect[0] + padded_rect[2], mid_y)))

    return arm_rects, arm_segments


def main() -> None:
    """Run the head-dodging game."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access webcam.")
        return
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from webcam.")
        return
    frame_h, frame_w = frame.shape[:2]

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
    spawn_interval = 1.5
    last_spawn_time = time.time()
    score = 0
    game_over = False
    last_head_rect: Optional[Rect] = None
    last_arm_rects: List[Rect] = []
    last_arm_segments: List[ArmSegment] = []
    last_arm_seen_at = 0.0
    previous_gray: Optional[np.ndarray] = None
    previous_frame_time = time.perf_counter() - TARGET_FRAME_TIME

    while True:
        frame_started_at = time.perf_counter()
        delta_time = frame_started_at - previous_frame_time
        previous_frame_time = frame_started_at

        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pose_results = None
        if pose_tracker is not None:
            if pose_tracker_backend == "solutions":
                pose_results = pose_tracker.process(rgb_frame)
            elif pose_tracker_backend == "tasks":
                frame_timestamp_ms = int(frame_started_at * 1000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                pose_results = pose_tracker.detect_for_video(mp_image, frame_timestamp_ms)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        head_rect: Optional[Rect] = None
        if len(faces) > 0:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            head_rect = tuple(int(value) for value in largest_face)
            last_head_rect = head_rect

        target_head_rect = head_rect if head_rect is not None else last_head_rect
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

        if not game_over and target_head_rect is not None and (now - last_spawn_time > spawn_interval):
            if random.random() < 0.55:
                attacks.append(create_jab_attack(target_head_rect, frame_w, frame_h, now))
            else:
                attacks.append(create_hook_attack(target_head_rect, frame_w, frame_h, now))
            last_spawn_time = now

        if not game_over:
            remaining_attacks: List[Attack] = []
            for attack in attacks:
                attack.update(now, delta_time)
                hit_rect = attack.get_hit_rect()

                if isinstance(attack, HookAttack) and hit_rect is not None:
                    if any(intersects(arm_rect, hit_rect) for arm_rect in arm_rects):
                        continue

                if head_rect is not None and hit_rect is not None and intersects(head_rect, hit_rect):
                    game_over = True

                if attack.is_finished(frame_w, frame_h):
                    score += 1
                else:
                    remaining_attacks.append(attack)
            attacks = remaining_attacks

        if head_rect is not None:
            x, y, w, h = head_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for arm_start, arm_end in arm_segments:
            cv2.line(frame, arm_start, arm_end, (255, 255, 0), ARM_DRAW_THICKNESS)
            cv2.circle(frame, arm_start, ARM_DRAW_THICKNESS + 2, (255, 255, 0), -1)
            cv2.circle(frame, arm_end, ARM_DRAW_THICKNESS + 2, (255, 255, 0), -1)

        for attack in attacks:
            attack.draw(frame, now)

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
            "Yellow = jab/hook warning | Orange = live hook | Cyan guard blocks hooks",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        if pose_tracker is None:
            cv2.putText(
                frame,
                pose_status_text or "MediaPipe pose unavailable: hook blocking off",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        if game_over:
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

        cv2.imshow(WINDOW_NAME, frame)

        frame_elapsed = time.perf_counter() - frame_started_at
        remaining_frame_time = TARGET_FRAME_TIME - frame_elapsed
        if remaining_frame_time > 0:
            time.sleep(remaining_frame_time)

        key = cv2.waitKey(1) & 0xFF
        if mouse_state["restart_requested"]:
            attacks = []
            last_spawn_time = time.time()
            score = 0
            game_over = False
            last_head_rect = head_rect
            last_arm_rects = []
            last_arm_segments = []
            last_arm_seen_at = 0.0
            previous_gray = None
            mouse_state["restart_requested"] = False
        if key == ord("q"):
            break

        previous_gray = cv2.GaussianBlur(gray, (7, 7), 0)

    cap.release()
    if pose_tracker is not None:
        pose_tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
