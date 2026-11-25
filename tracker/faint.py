# -*- coding: utf-8 -*-
"""微弱亮度抬升定位器:渐隐亮目标的专用外观通道。

物理依据:亮目标变淡(α 下降)时,它所在区域仍比周边背景「亮一点」,
这个抬升量与 α 成正比 —— 因此:
- 位置 = 门控区域内「亮度抬升图」的峰值;
- 置信度 = 当前抬升 / 初始抬升(随透明度单调衰减,正好可当置信度)。

通道选择(初始化时按信噪比自动挑选):
- whiteness:min(B,G,R)。对「中性白/灰目标 + 暖色/彩色背景」判别力最强
  (冒险岛沙色背景饱和度高、白度低,白色方块白度极高);
- gray:普通灰度。对「亮目标 + 暗背景」或「暗目标 + 亮背景」有效。

参考实现细节:用 boxFilter 尺寸 TW(目标尺寸)与 3×TW 的均值差作为
「中心相对邻域的抬升」,对整体光照变化稳健;loc 直接是像素坐标。
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class FaintBrightTracker:
    """门控的亮度抬升峰值定位。"""

    def __init__(self, min_snr: float = 1.2, min_lift: float = 8.0) -> None:
        self.min_snr = float(min_snr)
        self.min_lift = float(min_lift)

        self.reliable = False
        self.channel: str = ""
        self.scale: float = 0.0            # 初始抬升(可正可负,取绝对值)
        self.min_strength: float = 0.20    # 自适应最小强度 = max(0.2, 2×噪比)
        self._size: int = 64               # 抬升窗口尺寸(≈目标尺寸)
        self._big: int = 192               # 邻域窗口尺寸

    # ------------------------------------------------------------------ 初始化

    def init(self, frame: np.ndarray, bbox) -> bool:
        x, y, w, h = (int(round(v)) for v in bbox)
        self._size = max(24, int(min(w, h)))
        self._big = self._size * 3

        best = ("", 0.0, 0.0)  # (channel, lift@ROI, snr)
        for name, ch in (
            ("whiteness", self._whiteness(frame)),
            ("gray", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)),
        ):
            # 关键:scale 取「用户框选位置」的抬升,而不是全图峰值 ——
            # 初始帧附近可能有更亮的干扰(如倒计时数字),不能被它们带偏。
            lift, noise = self._lift_at(ch, x, y, w, h)
            snr = abs(lift) / max(noise, 1e-3)
            if abs(lift) >= self.min_lift and snr > best[2]:
                best = (name, lift, snr)

        self.channel, lift, snr = best
        self.reliable = bool(self.channel) and snr >= self.min_snr
        if self.reliable:
            self.scale = abs(lift)
            # 自适应门槛:背景抬升噪声越大(信噪比越低),要求的目标证据越强。
            # 噪比 = noise/scale = 1/snr。门槛取 3 倍噪比:抬升落入噪声带的
            # 候选一律弃权(交给运动通道),避免被背景亮斑带偏。
            self.min_strength = float(max(0.20, min(0.8, 3.0 / max(snr, 1e-3))))
        return self.reliable

    def reset(self) -> None:
        self.reliable = False
        self.channel = ""
        self.scale = 0.0

    # ------------------------------------------------------------------ 定位

    def locate(
        self,
        frame: np.ndarray,
        pred_center: Tuple[float, float],
        gate: float,
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        """在预测点附近 gate 半径内找抬升峰值。

        返回 ((x, y), strength):strength = 当前抬升 / 初始抬升(可 >1)。
        未初始化或搜索窗非法时返回 (None, 0)。
        """
        if not self.reliable or self.scale <= 0:
            return None, 0.0

        H, W = frame.shape[:2]
        px, py = pred_center
        m = int(gate + self._size)
        x0, x1 = max(0, int(px - m)), min(W, int(px + m))
        y0, y1 = max(0, int(py - m)), min(H, int(py + m))
        if x1 - x0 < self._size or y1 - y0 < self._size:
            return None, 0.0

        if self.channel == "whiteness":
            ch = self._whiteness(frame)
        else:
            ch = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        roi = ch[y0:y1, x0:x1]
        inner = cv2.boxFilter(roi, -1, (self._size, self._size), normalize=True)
        outer = cv2.boxFilter(roi, -1, (self._big, self._big), normalize=True)
        lift = inner - outer

        _, peak, _, loc = cv2.minMaxLoc(lift)
        if peak * np.sign(self.scale) < 0:  # 极性反了(目标亮/暗与初始化不符)
            return None, 0.0
        cx, cy = loc[0] + x0, loc[1] + y0
        strength = float(peak) / self.scale
        return (int(cx), int(cy)), abs(strength)

    # ------------------------------------------------------------------ 内部工具

    @staticmethod
    def _whiteness(frame: np.ndarray) -> np.ndarray:
        """中性白度:min(B,G,R),对彩色背景上的白色目标判别力最强。"""
        b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        return np.minimum(np.minimum(b, g), r).astype(np.float32)

    def _lift_at(self, ch: np.ndarray, x: int, y: int, w: int, h: int):
        """(用户 ROI 处的抬升, 背景噪声水平):初始帧计算,用于选通道。"""
        H, W = ch.shape[:2]
        cx, cy = x + w // 2, y + h // 2
        m = self._big
        x0, x1 = max(0, cx - m), min(W, cx + m)
        y0, y1 = max(0, cy - m), min(H, cy + m)
        if x1 - x0 < self._big or y1 - y0 < self._big:
            return 0.0, 1e3
        roi = ch[y0:y1, x0:x1]
        inner = cv2.boxFilter(roi, -1, (self._size, self._size), normalize=True)
        outer = cv2.boxFilter(roi, -1, (self._big, self._big), normalize=True)
        lift = inner - outer
        # ROI 中心处的抬升(而不是全图峰值):用户指哪,就以哪为参考
        lx, ly = cx - x0, cy - y0
        lx = min(max(lx, 0), lift.shape[1] - 1)
        ly = min(max(ly, 0), lift.shape[0] - 1)
        val = float(lift[ly, lx])
        # 噪声水平:去掉 ROI 邻域后的抬升图标准差
        masked = lift.copy()
        yy, xx = np.ogrid[: lift.shape[0], : lift.shape[1]]
        near = (xx - lx) ** 2 + (yy - ly) ** 2 < (self._size * 1.5) ** 2
        masked[near] = np.nan
        noise = float(np.nanstd(masked)) if not np.all(np.isnan(masked)) else 1e3
        return val, max(noise, 1e-3)
