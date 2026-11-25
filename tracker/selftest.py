# -*- coding: utf-8 -*-
"""无 GUI 的端到端自检:模拟“选框→逐帧追踪→丢失→重选”全流程。

场景 A:困难场景(随机移动/形态变化/半透明/相似干扰物)下验证追踪精度;
场景 B:目标消失后验证 LOST 判定与重新框选恢复。
(框在“外观相似的干扰物”上时,外观信号无法与真目标区分,需人工
 逐帧确认并按 F3 重选——这是设计内的交互,不在自检断言范围。)

用法: python run_tracker.py --selftest
"""

from __future__ import annotations

from typing import List

import numpy as np

from .core import TrackStatus, TargetTracker
from .synthetic import FadeRectScene, SyntheticScene


def _run_track(tracker: TargetTracker, scene: SyntheticScene, frames: int):
    errors: List[float] = []
    lost_frames = 0
    statuses = {"TRACKING": 0, "LOW_CONF": 0, "LOST": 0, "IDLE": 0}
    lost_at = None

    for _ in range(frames):
        frame, gt_bbox = scene.step()
        result = tracker.update(frame)
        statuses[result.status.value] += 1

        if gt_bbox is not None:
            if result.status == TrackStatus.LOST:
                lost_frames += 1
                continue
            gx = gt_bbox[0] + gt_bbox[2] / 2
            gy = gt_bbox[1] + gt_bbox[3] / 2
            if result.center is not None:
                errors.append(
                    ((result.center[0] - gx) ** 2 + (result.center[1] - gy) ** 2) ** 0.5
                )
        elif result.status == TrackStatus.LOST and lost_at is None:
            lost_at = tracker._miss  # 目标消失后约第几帧判定丢失

    mean_err = float(np.mean(errors)) if errors else float("inf")
    rate = len(errors) / frames if frames else 0.0
    return mean_err, rate, lost_frames, statuses, lost_at


def run_selftest() -> int:
    print("=" * 64)
    print("自检:合成困难场景(随机移动/形态变化/半透明/相似干扰物)")
    print("=" * 64)

    # ---- 场景 A:困难场景下的追踪精度(目标全程可见) ----
    print("\n[场景A] 5 个相似干扰物 + 形变 + 半透明,目标全程可见")
    tracker = TargetTracker(tracker_type="CSRT", snap=True)
    scene = SyntheticScene(seed=7, n_distractors=5)
    frame, bbox = scene.step()
    if not tracker.init(frame, bbox):
        print("FAIL: 追踪器初始化失败")
        return 1
    print(f"[init] tracker={tracker.tracker_name} bbox={bbox}")
    mean_err, rate, lost_frames, statuses, _ = _run_track(tracker, scene, 90)
    print(f"[统计] 平均中心误差={mean_err:.1f}px  有效定位率={rate:.0%}  "
          f"误判丢失帧={lost_frames}  状态分布={statuses}")

    # ---- 场景 B:目标消失 → LOST 判定 → 重新框选恢复 ----
    print("\n[场景B] 无干扰物,第 30 帧后目标消失")
    tracker_b = TargetTracker(tracker_type="CSRT", snap=True)
    scene_b = SyntheticScene(seed=15, n_distractors=0, hide_target_after=30)
    frame, bbox = scene_b.step()
    tracker_b.init(frame, bbox)
    _, _, _, statuses_b, lost_at = _run_track(tracker_b, scene_b, 60)
    print(f"[统计] 状态分布={statuses_b}  LOST 判定出现在目标消失后约第 {lost_at} 帧")

    frame_r, bbox_r = SyntheticScene(seed=14, n_distractors=0).step()
    reinit_ok = tracker_b.init(frame_r, bbox_r)
    result = tracker_b.update(SyntheticScene(seed=14, n_distractors=0).step()[0])
    reinit_ok = reinit_ok and result.status in (TrackStatus.TRACKING, TrackStatus.LOW_CONF)
    print(f"[重选] 重新框选并恢复追踪: {'OK' if reinit_ok else 'FAIL'}")

    # ---- 场景 C:白色矩形渐隐 → 只剩移动水波纹,运动方向追踪 ----
    print("\n[场景C] 白色矩形旋转 → 渐隐融入水纹背景 → 近乎不可见随机移动")
    tracker_c = TargetTracker(snap=True, motion=True)
    scene_c = FadeRectScene(seed=3)
    frame, bbox = scene_c.step()
    tracker_c.init(frame, bbox)
    fade_errs, lost_c = [], 0
    for _ in range(200):
        frame, gt = scene_c.step()
        result = tracker_c.update(frame)
        if result.status == TrackStatus.LOST:
            lost_c += 1
        gx, gy = gt[0] + gt[2] / 2, gt[1] + gt[3] / 2
        if result.center is not None and not scene_c.visible:
            fade_errs.append(
                ((result.center[0] - gx) ** 2 + (result.center[1] - gy) ** 2) ** 0.5
            )
    mean_c = float(np.mean(fade_errs)) if fade_errs else float("inf")
    print(f"[统计] 不可见阶段平均中心误差={mean_c:.1f}px  LOST帧={lost_c}/200")

    checks = [
        ("使用 CSRT 追踪器", tracker.tracker_name == "CSRT"),
        ("[A] 有效定位率 >= 90%", rate >= 0.90),
        ("[A] 平均中心误差 < 25px", mean_err < 25.0),
        ("[A] 目标可见时无误判丢失", lost_frames <= 3),
        ("[B] 目标消失后能判定 LOST", statuses_b["LOST"] > 0 and lost_at is not None and lost_at <= 25),
        ("[B] 重新框选可恢复追踪", reinit_ok),
        ("[C] 渐隐目标不丢失", lost_c == 0),
        ("[C] 不可见阶段平均误差 < 15px", mean_c < 15.0),
    ]
    print()
    all_ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        all_ok = all_ok and passed
    print("=" * 64)
    print("自检结果:", "全部通过 ✔" if all_ok else "存在失败项 ✘")
    return 0 if all_ok else 1
