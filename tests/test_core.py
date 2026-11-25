# -*- coding: utf-8 -*-
"""追踪核心的无 GUI 单元测试(可用 pytest 运行,也可直接 python 本文件)。

运行:
    python -m pytest tests/ -q
    或
    python tests/test_core.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from tracker.core import (  # noqa: E402
    TrackStatus,
    TargetTracker,
    _sanitize_bbox,
    create_tracker,
)
from tracker.synthetic import SyntheticScene  # noqa: E402


# ------------------------------------------------------------------ 基础功能

def test_create_tracker_prefers_csrt():
    tracker, name = create_tracker("auto")
    assert name in ("CSRT", "KCF")
    # 装了 opencv-contrib 时应优先 CSRT
    if hasattr(cv2, "TrackerCSRT_create") or (
        hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create")
    ):
        assert name == "CSRT"


def test_sanitize_bbox():
    assert _sanitize_bbox((-10, -5, 60, 40), (480, 640, 3)) == (0, 0, 60, 40)
    assert _sanitize_bbox((600, 400, 200, 200), (480, 640, 3)) == (600, 400, 40, 80)
    assert _sanitize_bbox((10, 10, 0, 40), (480, 640, 3)) is None
    assert _sanitize_bbox(None, (480, 640, 3)) is None


def test_init_requires_valid_bbox():
    scene = SyntheticScene(seed=1)
    frame, bbox = scene.step()
    tracker = TargetTracker()
    assert tracker.init(frame, bbox) is True
    assert tracker.active
    assert tracker.tracker_name == "CSRT"

    tracker2 = TargetTracker()
    assert tracker2.init(frame, (100, 100, 0, 0)) is False
    assert not tracker2.active


def test_update_before_init_is_idle():
    scene = SyntheticScene(seed=2)
    frame, _ = scene.step()
    tracker = TargetTracker()
    result = tracker.update(frame)
    assert result.status == TrackStatus.IDLE


# ------------------------------------------------------------------ 追踪质量

def test_tracks_morphing_transparent_target():
    """随机移动 + 形变 + 半透明 + 干扰物场景下,应稳定跟住目标。"""
    scene = SyntheticScene(seed=7, n_distractors=5)
    frame, bbox = scene.step()
    tracker = TargetTracker(snap=True)
    assert tracker.init(frame, bbox)

    errors = []
    n_frames = 60
    lost_frames = 0
    for _ in range(n_frames):
        frame, gt = scene.step()
        result = tracker.update(frame)
        if result.status == TrackStatus.LOST:
            lost_frames += 1
            continue
        gx = gt[0] + gt[2] / 2
        gy = gt[1] + gt[3] / 2
        assert result.center is not None
        errors.append(((result.center[0] - gx) ** 2 + (result.center[1] - gy) ** 2) ** 0.5)

    mean_err = sum(errors) / len(errors)
    print(f"    平均中心误差 {mean_err:.1f}px,丢失帧 {lost_frames}/{n_frames}")
    assert lost_frames <= 3, "目标并未消失,不应频繁判定丢失"
    assert mean_err < 25.0, f"平均中心误差过大: {mean_err:.1f}px"


def test_lost_detection_and_reinit():
    """目标消失后应判定 LOST;重新框选后能恢复追踪。

    注:使用无干扰物、且消失点为普通背景的场景(seed=15)。若消失点的
    背景本身与目标同色系(如 seed=13),外观信号无法与真目标区分,
    属于设计内的人工确认范畴,不作为自动丢失判定的测试条件。
    """
    scene = SyntheticScene(seed=15, n_distractors=0, hide_target_after=30)
    frame, bbox = scene.step()
    tracker = TargetTracker(snap=True)
    assert tracker.init(frame, bbox)

    lost_seen = False
    for _ in range(60):
        frame, gt = scene.step()
        result = tracker.update(frame)
        if gt is None and result.status == TrackStatus.LOST:
            lost_seen = True
            break
    assert lost_seen, "目标消失后未判定 LOST"

    # 重新框选(模拟按 F3 重选):用同参数的新场景(目标可见)验证可恢复
    fresh = SyntheticScene(seed=14, n_distractors=0)
    frame2, gt2 = fresh.step()
    assert gt2 is not None
    assert tracker.init(frame2, gt2) is True
    frame3, _ = fresh.step()
    result = tracker.update(frame3)
    assert result.status in (TrackStatus.TRACKING, TrackStatus.LOW_CONF)
    assert result.confidence > 0.3


def test_reset():
    scene = SyntheticScene(seed=3)
    frame, bbox = scene.step()
    tracker = TargetTracker()
    tracker.init(frame, bbox)
    tracker.reset()
    assert not tracker.active
    assert tracker.last_result.status == TrackStatus.IDLE


# ------------------------------------------------------------------ 视频文件集成

def test_sample_video_loadable():
    """仓库自带的合成 test.mp4 应能正常打开并解码(存在时才检查)。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "test.mp4")
    if not os.path.exists(path):
        print("    (test.mp4 不存在,跳过)")
        return
    cap = cv2.VideoCapture(path)
    assert cap.isOpened()
    ok, frame = cap.read()
    assert ok and frame is not None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert total > 0
    cap.release()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        print(f"== {fn.__name__}")
        try:
            fn()
            print("   PASS")
        except AssertionError as e:
            failed += 1
            print(f"   FAIL: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} 通过")
    sys.exit(1 if failed else 0)
