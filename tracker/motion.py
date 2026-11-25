# -*- coding: utf-8 -*-
"""运动能量追踪器:目标与背景融合后的最后防线。

原理(模拟人类“根据移动方向持续追踪”的策略):
- 目标颜色/外观与背景融合后,模板、直方图全部失效;但只要它还在移动,
  相邻帧差分就会在它的(残影)位置留下能量;
- 用「帧差能量图 + 衰减残影」累积运动证据,静止背景保持零能量,
  背景自带的静态水波纹也不会产生能量;
- 用「预测门控 + 方向一致性」做数据关联:只在预测位置附近的圆形门内
  找能量质心,且位移方向要与当前速度估计一致——这正是人眼持续追踪
  所用的先验,能过滤掉远处偶发的噪声团;
- 内置全局运动补偿(phaseCorrelate),轻微的整体漂移不会污染质心。

输入是灰度图;目标近乎不可见(对比度只有几个灰度级)时,先高斯平滑
抑制噪声,再累积多帧能量,把弱信号从噪声里“熬”出来。
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

Vec2 = Tuple[float, float]


class MotionTracker:
    """帧差能量 + 方向门控的运动定位器。"""

    def __init__(
        self,
        blur: int = 7,
        trail_decay: float = 0.55,
        trail_weight: float = 0.45,
        min_strength: float = 1.6,
        dir_cos: float = 0.1,
        vel_alpha: float = 0.35,
        global_motion: bool = True,
        strong_strength: float = 10.0,
    ) -> None:
        self.blur = int(blur) if int(blur) % 2 == 1 else int(blur) + 1
        self.trail_decay = float(trail_decay)
        self.trail_weight = float(trail_weight)
        self.min_strength = float(min_strength)
        self.dir_cos = float(dir_cos)
        self.vel_alpha = float(vel_alpha)
        self.global_motion = bool(global_motion)
        # 强度超过该值的候选几乎必是真目标(背景噪声远达不到),
        # 跳过方向一致性检查——目标急转弯时不应被自己的先验拒绝。
        self.strong_strength = float(strong_strength)

        self._prev: Optional[np.ndarray] = None       # 上一帧灰度(已平滑)
        self._trail: Optional[np.ndarray] = None      # 衰减残影能量图
        self._history: deque = deque(maxlen=4)        # 最近几帧灰度,用于多基线帧差
        self.vel = np.zeros(2)                        # 速度估计 (vx, vy)
        self.last_center: Optional[np.ndarray] = None # 上次报告的目标中心
        self.strength = 0.0                           # 最近一次定位的能量强度
        self.frames = 0

    def reset(self) -> None:
        self._prev = None
        self._trail = None
        self._history.clear()
        self.vel = np.zeros(2)
        self.last_center = None
        self.strength = 0.0
        self.frames = 0

    # ------------------------------------------------------------------ 主流程

    def locate(
        self,
        gray: np.ndarray,
        pred_center: Optional[Vec2],
        gate: float,
        size_hint: float = 30.0,
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        """在预测点附近 gate 半径内找运动质心。

        返回 (center|None, strength):center 为质心像素坐标,strength 是
        能量强度(背景噪声水平归一后,>1 视为有效运动证据)。
        """
        smooth = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)
        if self._prev is None or self._prev.shape != smooth.shape:
            self._prev = smooth
            self._history.append(smooth)
            return None, 0.0

        # 全局运动补偿(轻微整体漂移/抖动)
        dx = dy = 0.0
        if self.global_motion:
            try:
                (gx, gy), _ = cv2.phaseCorrelate(
                    self._prev.astype(np.float32), smooth.astype(np.float32)
                )
                if abs(gx) < 8 and abs(gy) < 8:  # 超过此限视为异常,不做补偿
                    dx, dy = float(gx), float(gy)
            except cv2.error:
                pass

        # 多时间基线帧差:慢速移动时 1 帧基线的位移不足 1 像素,
        # 能量极弱;取 k=1/2/3 帧差的最大值,把慢速运动的能量放大。
        diffs = []
        hist = list(self._history)
        for k in (1, 2, 3):
            if len(hist) >= k:
                d = cv2.GaussianBlur(cv2.absdiff(smooth, hist[-k]), (self.blur, self.blur), 0)
                diffs.append(d)
        diff = diffs[0] if diffs else cv2.GaussianBlur(
            cv2.absdiff(smooth, self._prev), (self.blur, self.blur), 0
        )
        for d in diffs[1:]:
            diff = np.maximum(diff, d)
        if dx or dy:
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            diff = cv2.warpAffine(diff, M, (diff.shape[1], diff.shape[0]))

        self._prev = smooth
        self._history.append(smooth)
        self.frames += 1

        # 能量 = 当前帧差 + 衰减残影(把弱信号在时间上累积)
        if self._trail is None or self._trail.shape != diff.shape:
            self._trail = diff.copy()
        else:
            self._trail = np.maximum(self._trail * self.trail_decay, diff)
        energy = diff.astype(np.float32) + self.trail_weight * self._trail

        H, W = energy.shape
        # 搜索门:预测点附近的圆形区域;没有预测时全图(冷启动)。
        mask = np.ones_like(energy)
        if pred_center is not None:
            yy, xx = np.mgrid[0:H, 0:W]
            mask = ((xx - pred_center[0]) ** 2 + (yy - pred_center[1]) ** 2) <= gate * gate
            mask = mask.astype(np.float32)

        gated = energy * mask
        if gated.max() <= 0:
            self.strength = 0.0
            return None, 0.0

        # 自适应阈值:门内能量的分位数 + 绝对下限
        vals = gated[mask > 0]
        thr = max(self.min_strength, float(np.percentile(vals, 97)))
        hit = (gated > thr).astype(np.float32)
        total = float(hit.sum())
        if total < 6:  # 命中像素太少,不可信
            self.strength = 0.0
            return None, 0.0

        ys, xs = np.nonzero(hit > 0)
        w = gated[ys, xs]
        cx = float(np.average(xs, weights=w))
        cy = float(np.average(ys, weights=w))

        center = np.array([cx, cy])
        strength = self._norm_strength(energy, center, size_hint)

        # 方向一致性:位移方向应与速度估计一致(人眼追踪的先验)。
        # 但证据极强时跳过该检查(急转弯不该被先验拒绝);
        # 被拒帧衰减速度估计,连续被拒时门自动放宽以便重新捕获。
        if self.last_center is not None and strength < self.strong_strength:
            step = center - self.last_center
            speed = float(np.linalg.norm(self.vel))
            if speed > 0.8 and np.linalg.norm(step) > 1e-3:
                cos = float(np.dot(step, self.vel) / (np.linalg.norm(step) * speed))
                if cos < self.dir_cos:
                    self.vel *= 0.6
                    self.strength = 0.0
                    return None, 0.0

        # 更新速度与状态
        if self.last_center is not None:
            inst = center - self.last_center
            self.vel = (1 - self.vel_alpha) * self.vel + self.vel_alpha * inst
        self.last_center = center
        self.strength = strength
        return (int(round(cx)), int(round(cy))), strength

    # ------------------------------------------------------------------ 内部工具

    def _norm_strength(self, energy: np.ndarray, center: np.ndarray, size_hint: float) -> float:
        """目标邻域能量 / 全图背景能量水平,得到 ~1 以上的强度。"""
        H, W = energy.shape
        x0 = max(0, int(center[0] - size_hint))
        x1 = min(W, int(center[0] + size_hint))
        y0 = max(0, int(center[1] - size_hint))
        y1 = min(H, int(center[1] + size_hint))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        local = float(energy[y0:y1, x0:x1].mean())
        other = energy.copy()
        other[y0:y1, x0:x1] = 0.0
        bg = float(other.mean())
        return local / max(bg + 1e-3, 1e-3)
