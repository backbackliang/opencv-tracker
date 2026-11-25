# -*- coding: utf-8 -*-
"""画面叠加绘制(HUD):追踪框、中心坐标、置信度、状态与按键提示。

注:OpenCV 的 putText 不支持中文,界面文字用英文;控制台输出为中文。
"""

from __future__ import annotations

from typing import List, Optional

import cv2

from .core import TrackResult, TrackStatus

FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOR_TRACK = (60, 220, 60)     # 绿:正常追踪
COLOR_WARN = (60, 200, 255)     # 黄橙:低置信
COLOR_LOST = (60, 60, 230)      # 红:丢失
COLOR_BOX_LOST = (150, 150, 150)
COLOR_PANEL = (25, 25, 25)

STATUS_COLOR = {
    TrackStatus.TRACKING: COLOR_TRACK,
    TrackStatus.LOW_CONF: COLOR_WARN,
    TrackStatus.LOST: COLOR_LOST,
    TrackStatus.IDLE: COLOR_BOX_LOST,
}


def _text(frame, s, org, scale, color, thickness=1, shadow=True) -> None:
    if shadow:
        cv2.putText(frame, s, (org[0] + 1, org[1] + 1), FONT, scale, (0, 0, 0),
                    thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, s, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_target(frame, result: TrackResult, tracker_name: str = "") -> None:
    """绘制追踪框,并在框上方标注中心坐标 (X, Y) 与置信度/状态。"""
    status = result.status
    if status == TrackStatus.IDLE:
        return

    if result.bbox is not None:
        x, y, w, h = result.bbox
        color = STATUS_COLOR[status]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # LOST 时坐标已过期,不显示框上标签,避免与中央提示重叠。
        if status != TrackStatus.LOST and result.center is not None:
            label_y = y - 8
            if label_y < 16:
                label_y = y + h + 18
            conf = result.confidence
            label = f"({result.center[0]},{result.center[1]}) conf={conf:.2f} {status.value}"
            if result.note == "snap":
                label += " [snap]"
            _text(frame, label, (x, label_y), 0.5, color, 1)

    if status == TrackStatus.LOST:
        H, W = frame.shape[:2]
        _text(frame, "TARGET LOST", (W // 2 - 130, H // 2), 1.1, COLOR_LOST, 2)
        _text(frame, "Press F3 or R to re-select target",
              (W // 2 - 190, H // 2 + 30), 0.6, (255, 255, 255), 1)


def draw_status_bar(frame, frame_idx: int, tracker_name: str,
                    result: TrackResult, playing: bool, ended: bool) -> None:
    """左上角信息面板:帧号、模式、按键提示。"""
    H, W = frame.shape[:2]

    if ended:
        hint = "Video ended - R: restart from frame 0   Q: quit"
    elif result.status == TrackStatus.LOST:
        hint = "TARGET LOST - F3 / R: re-select target (drag, ENTER to confirm)"
    elif result.status == TrackStatus.IDLE:
        hint = "F3 / R: select target (left-drag, ENTER to confirm)"
    elif playing:
        hint = "P: pause   SPACE: step frame   F3 / R: re-select   Q: quit"
    else:
        hint = "SPACE: next frame   P: auto-play   F3 / R: re-select   Q: quit"

    mode = "PLAY" if playing else "STEP"
    line1 = f"frame {frame_idx}  |  tracker {tracker_name}  |  {mode}"
    lines: List[str] = [line1, hint]

    panel_h = 14 + 22 * len(lines)
    panel = frame[0:panel_h, 0:W]
    dark = (panel * 0.35).astype(frame.dtype)
    frame[0:panel_h, 0:W] = cv2.addWeighted(dark, 0.8, panel, 0.2, 0)

    for i, s in enumerate(lines):
        _text(frame, s, (10, 22 + 22 * i), 0.55, (235, 235, 235), 1)


def draw_frame_overlay(frame, frame_idx: int, tracker_name: str,
                       result: TrackResult, playing: bool, ended: bool) -> None:
    """整帧 HUD:目标框 + 状态栏。"""
    draw_target(frame, result, tracker_name)
    draw_status_bar(frame, frame_idx, tracker_name, result, playing, ended)
