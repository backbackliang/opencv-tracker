# -*- coding: utf-8 -*-
"""微弱目标定位器:亮度抬升 + 「平静洞」联合证据(渐隐目标专用)。

物理依据(alpha 合成 dst = α·目标色 + (1-α)·背景):
- 亮度抬升:目标区域比周边背景亮 α·(目标色-背景色);
- 平静洞:背景自身在动(滚动/闪烁),目标区域内的一切时间变化都被
  (1-α) 压低 —— 运动能量图上,目标是一个「比周边平静」的洞。

两条线索的虚警源不同(亮斑/倒计时数字 vs 低纹理静区),联合起来可以
在压住虚警的同时采用更低门槛,从而追到更深的透明度。检测全部以
z 分数进行(各图做 MAD 稳健归一化),直接取联合 z 峰值作为候选。

接受判定配套四重防线,防止深渐隐阶段的弱证据被干扰抢锁:
- 联合 z 门槛:连续追踪 z_alive(2.2),冷启动/复活 z_cold(3.2);
- 强度下限:候选抬升相对初始 scale 不得低于 cold_min_strength
  (默认 0.35;LOST 后复活时由调用方收紧到 0.9);
- 预测邻域:冷启动候选必须落在预测点 gate + 半个目标尺寸内 ——
  链断期间不允许远距离抢锁(长距离恢复由 LOST 后 coast 门扩大承担);
- 跳变拒绝:瞬移(>500px)一律拒绝;连续追踪中的大跳按速度上限拒绝,
  但联合证据极强(z≥6 且强度≥0.35)时豁免 —— 真目标也可能突然加速。

LOST 后按速度惯性滑行(coast),门槛切换为更严的冷启动值,门随时间
扩大;一旦联合证据重新出现(且位置合理)立即恢复锁定。
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def _zmap(map_: np.ndarray, sigma_floor: float = 1.0) -> np.ndarray:
    """稳健 z 分数图:中位数中心化,MAD 尺度归一化。"""
    med = float(np.median(map_))
    sigma = float(np.median(np.abs(map_ - med))) * 1.4826
    return (map_ - med) / max(sigma, sigma_floor)


class FaintBrightTracker:
    """联合「亮度抬升 + 平静洞」证据的渐隐目标定位器。"""

    def __init__(
        self,
        min_snr: float = 1.2,
        min_lift: float = 8.0,
        z_alive: float = 2.2,
        z_cold: float = 3.2,
    ) -> None:
        self.min_snr = float(min_snr)
        self.min_lift = float(min_lift)
        self.z_alive = float(z_alive)   # 连续追踪中的联合 z 门槛
        self.z_cold = float(z_cold)     # 冷启动/LOST 后复活的门槛
        self.cold_min_strength = 0.35   # 冷启动时候选抬升至少要有初始的 45%

        self.reliable = False
        self.channel: str = ""
        self.scale: float = 0.0            # 初始抬升(可正可负,取绝对值)

        self.vel = np.zeros(2)             # 目标速度估计(由连续接受的位置差更新)
        self.last_pos: Optional[Tuple[float, float]] = None
        self.since_accept = 999            # 距上次接受的帧数
        self.last_candidate_strength = 0.0 # 最近候选的抬升强度(相对初始)

        self._size: int = 64               # 抬升窗口尺寸(≈目标尺寸)
        self._big: int = 192               # 邻域窗口尺寸
        self._coast_cap = 25               # 滑行最多外推的帧数

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
        self.vel = np.zeros(2)
        self.last_pos = None
        self.since_accept = 999
        return self.reliable

    def reset(self) -> None:
        self.reliable = False
        self.channel = ""
        self.scale = 0.0
        self.vel = np.zeros(2)
        self.last_pos = None
        self.since_accept = 999

    # ------------------------------------------------------------------ 定位

    def locate(
        self,
        frame: np.ndarray,
        energy: Optional[np.ndarray],
        pred_center: Tuple[float, float],
        gate: float,
        cold_min_strength: float = 0.35,
        alive_min_strength: float = 0.0,
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        """在预测点附近 gate 半径内做联合证据定位。

        energy 为 MotionTracker.process() 输出的运动能量图(可为 None)。
        cold_min_strength:冷启动/复活时候选抬升的最低相对强度。
        返回 (center|None, conf):conf 由联合 z 峰值折算到 [0,1]。
        """
        if not self.reliable:
            return None, 0.0
        H, W = frame.shape[:2]
        px, py = pred_center
        m = int(gate + self._size)
        x0, x1 = max(0, int(px - m)), min(W, int(px + m))
        y0, y1 = max(0, int(py - m)), min(H, int(py + m))
        if x1 - x0 < self._size or y1 - y0 < self._size:
            return None, 0.0

        # 1) 亮度抬升 z 图
        if self.channel == "whiteness":
            ch = self._whiteness(frame)
        else:
            ch = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        roi = ch[y0:y1, x0:x1]
        inner = cv2.boxFilter(roi, -1, (self._size, self._size), normalize=True)
        outer = cv2.boxFilter(roi, -1, (self._big, self._big), normalize=True)
        lift = inner - outer
        if self.scale < 0:  # 初始为「暗目标」时取反,统一找正峰
            lift = -lift
        zl = _zmap(lift)

        # 2) 平静洞 z 图(背景在动而目标区被压低)
        zc = np.zeros_like(zl)
        if energy is not None:
            eroi = energy[y0:y1, x0:x1]
            ein = cv2.boxFilter(eroi, -1, (self._size, self._size), normalize=True)
            eout = cv2.boxFilter(eroi, -1, (self._big, self._big), normalize=True)
            calm = eout - ein  # 正值 = 比周边平静
            ambient = float(np.median(ein))
            if ambient >= 1.0:
                # 平静洞只作加分不作惩罚:目标自身移动时,其运动能量会
                # 掩盖平静洞(杆状负分),不能因此扣掉亮度抬升的证据。
                zc = np.maximum(_zmap(calm), 0.0)

        combined = zl + zc

        _, peak, _, loc = cv2.minMaxLoc(combined)
        pos = (float(loc[0] + x0), float(loc[1] + y0))
        lift_peak = float(lift[loc[1], loc[0]])
        strength = abs(lift_peak) / max(abs(self.scale), 1e-3)
        self.last_candidate_strength = strength

        # 4) 接受判定:连续追踪用低 z 门槛;冷启动/复活用高 z 门槛。
        #    强度下限 floor:连续追踪由调用方指定(默认 0,信任链上证据),
        #    冷启动 ≥ 初始强度的 0.35、LOST 后复活 ≥ 0.9(近乎恢复可见)
        #    —— 目标强度随透明度连续衰减,远低于下限的亮点(背景波纹环/
        #    亮斑/UI 亮字)一律拒绝。
        thr = self.z_alive if self.since_accept <= 3 else self.z_cold
        floor = alive_min_strength if self.since_accept <= 3 else cold_min_strength
        if peak < thr:
            self.since_accept += 1
            return None, 0.0
        if strength < floor:
            self.since_accept += 1
            return None, 0.0
        # 预测邻域约束(所有接受,含连续追踪):候选必须落在预测点
        # gate + 半个目标尺寸内。连续追踪时这是把 faint 约束在 CSRT/
        # motion 邻域的缰绳;LOST 后 coast 门随时间扩大,长距离恢复
        # 仍然允许 —— 远处的亮云/UI 因此永远够不着候选。
        d_pred = float(np.hypot(pos[0] - px, pos[1] - py))
        if d_pred > gate + self._size / 2:
            self.since_accept += 1
            return None, 0.0

        if self.last_pos is not None:
            step = np.array(pos) - np.array(self.last_pos)
            d = float(np.linalg.norm(step))
            # 瞬移拒绝:目标不可能跳走(远处亮区/UI 一律排除)
            if d > 500.0:
                self.since_accept += 1
                return None, 0.0
            if self.since_accept <= 3:
                # 连续追踪中突然大跳:可能是虚警抢锁(如「开始」闪现后的
                # 天空亮云);但联合证据极强(z≥6 且强度足)时豁免 ——
                # 真目标也可能突然加速。豁免的跳变距离同样封顶:
                # 加速的真目标仍在预测邻域量级,绝不可能一步数百像素。
                speed = float(np.linalg.norm(self.vel))
                limit = max(2.5 * speed + 24.0, 48.0)
                strong_cap = max(1.5 * gate, 120.0)
                strong = (
                    peak >= 6.0
                    and strength >= 0.35
                    and d <= strong_cap
                )
                if d > limit and not strong:
                    self.since_accept += 1
                    return None, 0.0
            self.vel = 0.6 * self.vel + 0.4 * step

        self.last_pos = pos
        self.since_accept = 0
        conf = min(1.0, float(peak) / 3.0)
        return (int(round(pos[0])), int(round(pos[1]))), conf

    def coast_center(self) -> Optional[Tuple[float, float]]:
        """LOST 后按速度惯性外推的目标可能位置(供扩大搜索门使用)。"""
        if self.last_pos is None:
            return None
        k = min(self.since_accept, self._coast_cap)
        return (
            self.last_pos[0] + self.vel[0] * k,
            self.last_pos[1] + self.vel[1] * k,
        )

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
