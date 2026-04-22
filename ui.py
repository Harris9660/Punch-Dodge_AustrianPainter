from typing import List, Optional

import cv2

from game_utils import ArmSegment, Rect
from settings import (
    ARM_DRAW_THICKNESS,
    AUTO_RESTART_DELAY,
    AUTO_RESTART_ENABLED,
    BOTH_MODE_STAGE_DURATION,
    SHOW_CAMERA_STATS,
    SHOW_FPS_COUNTER,
    SHOW_PERFORMANCE_BREAKDOWN,
    get_mode_label,
)


def draw_head_rect(frame, head_rect: Rect) -> None:
    """Draw the tracked head rectangle."""
    x, y, w, h = head_rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_arm_segments(frame, arm_segments: List[ArmSegment]) -> None:
    """Draw the blocking arm guides."""
    for arm_start, arm_end in arm_segments:
        cv2.line(frame, arm_start, arm_end, (255, 255, 0), ARM_DRAW_THICKNESS)
        cv2.circle(frame, arm_start, ARM_DRAW_THICKNESS + 2, (255, 255, 0), -1)
        cv2.circle(frame, arm_end, ARM_DRAW_THICKNESS + 2, (255, 255, 0), -1)


def draw_hud(
    frame,
    score: int,
    game_mode: str,
    current_stage: str,
    now: float,
    stage_started_at: float,
    pose_tracker_available: bool,
    pose_status_text: Optional[str],
    fps_display: float,
    camera_reported_width: int,
    camera_reported_height: int,
    camera_reported_fps: float,
    live_camera_fps_display: float,
    frame_ms_display: float,
    read_ms_display: float,
    pose_ms_display: float,
    face_ms_display: float,
) -> None:
    """Draw all status and instructional text on screen."""
    frame_h, frame_w = frame.shape[:2]

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

    if not pose_tracker_available:
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


def draw_game_over_overlay(frame, score: int, now: float, game_over_started_at: Optional[float]) -> Rect:
    """Draw the game-over screen and return the restart button rectangle."""
    frame_h, frame_w = frame.shape[:2]
    button_width = int(frame_w * 0.24)
    button_height = int(frame_h * 0.1)
    button_x = int((frame_w - button_width) / 2)
    button_y = int(frame_h * 0.68)
    restart_button_rect = (button_x, button_y, button_width, button_height)

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
    return restart_button_rect
