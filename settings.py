import os

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

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
