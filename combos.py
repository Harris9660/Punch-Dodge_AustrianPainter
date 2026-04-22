import random
from typing import List, Optional

from entities import Attack, HookAttack, JabAttack, PadTarget
from game_utils import Rect, rect_center
from settings import (
    COMBO_STEP_DELAY,
    HOOK_HEIGHT,
    HOOK_WARNING_DURATION,
    HOOK_WARNING_HEAD_GAP,
    HOOK_WARNING_WIDTH,
    HOOK_WIDTH,
    JAB_ACTIVE_DURATION,
    PAD_GROW_DURATION,
    PAD_HEAD_SIDE_OFFSET_RATIO,
    PAD_READY_RADIUS,
    PAD_READY_WINDOW,
    PAD_START_RADIUS,
    STRAIGHT_SIDE_HEAD_OFFSET_RATIO,
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
