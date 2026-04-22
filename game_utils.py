import math
from typing import Tuple

import cv2


Rect = Tuple[int, int, int, int]
Point = Tuple[int, int]
ArmSegment = Tuple[Point, Point]


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
