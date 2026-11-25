"""opencv-tracker: 基于 OpenCV CSRT/KCF 的交互式目标追踪工具包。"""

from .core import (
    BBox,
    TrackResult,
    TrackStatus,
    TargetTracker,
    create_tracker,
)

__all__ = [
    "BBox",
    "TrackResult",
    "TrackStatus",
    "TargetTracker",
    "create_tracker",
]
