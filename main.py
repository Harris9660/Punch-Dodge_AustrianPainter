import os
import sys
import time
from typing import List, Optional

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

from combos import create_attack_pattern, create_pad_target
from entities import Attack, HookAttack, JabAttack, PadTarget
from game_utils import ArmSegment, Rect, intersects, point_in_rect, segment_hits_circle, smooth_metric, smooth_rect
from settings import (
    ARM_MEMORY_SECONDS,
    AUTO_RESTART_DELAY,
    AUTO_RESTART_ENABLED,
    BOTH_MODE_STAGE_DURATION,
    FACE_MEMORY_SECONDS,
    FACE_SMOOTHING,
    FPS_SMOOTHING,
    FULLSCREEN_WINDOW,
    GAME_MODE,
    PAD_RESPAWN_DELAY,
    PAD_SCORE_POINTS,
    TARGET_FPS,
    TARGET_FRAME_TIME,
    WINDOW_NAME,
    mode_has_dodge,
    mode_has_pad_work,
    normalize_game_mode,
)
from ui import draw_arm_segments, draw_game_over_overlay, draw_head_rect, draw_hud
from vision import (
    LatestFrameCapture,
    configure_camera,
    create_face_cascade,
    create_pose_tracker,
    detect_arm_blocks,
    detect_arm_blocks_fallback,
    detect_head_rect,
    detect_pose_results,
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
    configure_camera(cap, TARGET_FPS)

    ret, initial_frame = cap.read()
    if not ret:
        print("Error: Unable to read from webcam.")
        cap.release()
        return

    frame_h, frame_w = initial_frame.shape[:2]
    capture_stream = LatestFrameCapture(cap, initial_frame)
    capture_stream.start()

    camera_reported_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_reported_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_reported_fps = cap.get(cv2.CAP_PROP_FPS)

    face_cascade = create_face_cascade()
    pose_tracker, pose_tracker_backend, pose_status_text = create_pose_tracker()

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
    previous_gray = None
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

    try:
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

            head_rect = None
            face_elapsed = 0.0
            if has_new_camera_frame:
                head_rect, face_elapsed = detect_head_rect(
                    gray,
                    face_cascade,
                    last_head_rect,
                    frame_w,
                    frame_h,
                    capture_frame_id,
                )

            if head_rect is not None:
                if last_head_rect is None:
                    last_head_rect = head_rect
                else:
                    last_head_rect = smooth_rect(last_head_rect, head_rect, FACE_SMOOTHING)
                last_head_seen_at = now
            elif last_head_rect is not None and (now - last_head_seen_at) > FACE_MEMORY_SECONDS:
                last_head_rect = None

            target_head_rect = last_head_rect

            if has_new_camera_frame:
                pose_results, pose_elapsed, last_pose_results = detect_pose_results(
                    frame,
                    pose_tracker,
                    pose_tracker_backend,
                    target_head_rect,
                    last_pose_results,
                    capture_frame_id,
                    frame_started_at,
                )
            else:
                pose_results = last_pose_results
                pose_elapsed = 0.0

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
            else:
                last_pose_results = None

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
                draw_head_rect(frame, target_head_rect)
            draw_arm_segments(frame, arm_segments)

            for attack in attacks:
                attack.draw(frame, now)

            if pad_work_enabled and not game_over and pad_target is not None:
                pad_target.draw(frame, now)

            if dodge_enabled and game_over:
                mouse_state["restart_button_rect"] = draw_game_over_overlay(
                    frame,
                    score,
                    now,
                    game_over_started_at,
                )
            else:
                mouse_state["restart_button_rect"] = None

            frame_elapsed = time.perf_counter() - frame_started_at
            frame_ms_display = smooth_metric(frame_ms_display, frame_elapsed * 1000.0, FPS_SMOOTHING)
            read_ms_display = smooth_metric(read_ms_display, read_elapsed * 1000.0, FPS_SMOOTHING)
            pose_ms_display = smooth_metric(pose_ms_display, pose_elapsed * 1000.0, FPS_SMOOTHING)
            face_ms_display = smooth_metric(face_ms_display, face_elapsed * 1000.0, FPS_SMOOTHING)

            draw_hud(
                frame,
                score,
                game_mode,
                current_stage,
                now,
                stage_started_at,
                pose_tracker is not None,
                pose_status_text,
                fps_display,
                camera_reported_width,
                camera_reported_height,
                camera_reported_fps,
                live_camera_fps_display,
                frame_ms_display,
                read_ms_display,
                pose_ms_display,
                face_ms_display,
            )

            cv2.imshow(WINDOW_NAME, frame)

            remaining_frame_time = TARGET_FRAME_TIME - frame_elapsed
            if remaining_frame_time > 0:
                time.sleep(remaining_frame_time)

            key = cv2.waitKey(1) & 0xFF
            if (
                dodge_enabled
                and AUTO_RESTART_ENABLED
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
    finally:
        capture_stream.stop()
        cap.release()
        if pose_tracker is not None:
            pose_tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
