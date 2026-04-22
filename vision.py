import os
import threading
import time
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
except Exception:
    mp_vision = None
    BaseOptions = None

from game_utils import ArmSegment, Point, Rect, clamp_rect, rect_center, smooth_metric
from settings import (
    ARM_BLOCK_LENGTH_PADDING,
    ARM_BLOCK_THICKNESS,
    ARM_GUARD_BOTTOM_MULTIPLIER,
    ARM_GUARD_TOP_MULTIPLIER,
    ARM_MIN_CONTOUR_AREA,
    ARM_PALM_EXTENSION_MIN_PIXELS,
    ARM_PALM_EXTENSION_MULTIPLIER,
    ARM_SEARCH_BOTTOM_MULTIPLIER,
    ARM_SEARCH_SIDE_MULTIPLIER,
    ARM_SEARCH_TOP_MULTIPLIER,
    CAMERA_BUFFER_SIZE,
    CAMERA_FRAME_HEIGHT,
    CAMERA_FRAME_WIDTH,
    CAMERA_USE_MJPG,
    FACE_DETECTION_EVERY_N_FRAMES,
    FACE_DETECTION_SCALE,
    FACE_SEARCH_SCALE,
    POSE_DETECTION_EVERY_N_FRAMES,
    POSE_INPUT_SCALE,
    POSE_LANDMARKER_MODEL_PATH,
    POSE_MIN_DETECTION_CONFIDENCE,
    POSE_MIN_PRESENCE_CONFIDENCE,
    POSE_MIN_TRACKING_CONFIDENCE,
    POSE_MODEL_COMPLEXITY,
    POSE_VISIBILITY_THRESHOLD,
    SKIN_YCRCB_LOWER,
    SKIN_YCRCB_UPPER,
)


MEDIAPIPE_SOLUTIONS_AVAILABLE = hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
MEDIAPIPE_TASKS_AVAILABLE = (
    mp_vision is not None
    and BaseOptions is not None
    and os.path.exists(POSE_LANDMARKER_MODEL_PATH)
)
POSE_LANDMARK = None
if MEDIAPIPE_SOLUTIONS_AVAILABLE:
    POSE_LANDMARK = mp.solutions.pose.PoseLandmark
elif MEDIAPIPE_TASKS_AVAILABLE:
    POSE_LANDMARK = mp_vision.PoseLandmark


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


def configure_camera(capture, target_fps: int) -> None:
    """Apply the requested camera settings before starting the game loop."""
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    if CAMERA_USE_MJPG and hasattr(cv2, "VideoWriter_fourcc"):
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, target_fps)


def create_face_cascade():
    """Load the face detector used by the game."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def create_pose_tracker():
    """Create the MediaPipe pose tracker when available."""
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

    return pose_tracker, pose_tracker_backend, pose_status_text


def detect_head_rect(
    gray,
    face_cascade,
    last_head_rect: Optional[Rect],
    frame_w: int,
    frame_h: int,
    capture_frame_id: int,
) -> Tuple[Optional[Rect], float]:
    """Detect the current head box, preferring a local search around the last face."""
    if capture_frame_id % max(1, FACE_DETECTION_EVERY_N_FRAMES) != 0:
        return None, 0.0

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
        if FACE_DETECTION_SCALE != 1.0:
            scaled_head_rect = (
                int(round(last_head_rect[0] * FACE_DETECTION_SCALE)),
                int(round(last_head_rect[1] * FACE_DETECTION_SCALE)),
                max(1, int(round(last_head_rect[2] * FACE_DETECTION_SCALE))),
                max(1, int(round(last_head_rect[3] * FACE_DETECTION_SCALE))),
            )
        else:
            scaled_head_rect = last_head_rect
        search_rect = get_face_search_rect(scaled_head_rect, face_input.shape[1], face_input.shape[0])

    largest_face = None
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

    if largest_face is None:
        return None, face_elapsed

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

    return clamp_rect(*head_rect, frame_w, frame_h), face_elapsed


def detect_pose_results(
    frame,
    pose_tracker,
    pose_tracker_backend: Optional[str],
    target_head_rect: Optional[Rect],
    last_pose_results,
    capture_frame_id: int,
    frame_started_at: float,
):
    """Run pose detection when needed and reuse the last result between frames."""
    pose_results = last_pose_results
    pose_elapsed = 0.0

    if pose_tracker is None:
        return None, pose_elapsed, None
    if target_head_rect is None:
        return None, pose_elapsed, None
    if last_pose_results is not None and capture_frame_id % max(1, POSE_DETECTION_EVERY_N_FRAMES) != 0:
        return pose_results, pose_elapsed, last_pose_results

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
    return pose_results, pose_elapsed, pose_results


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
    length = (dx * dx + dy * dy) ** 0.5
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
