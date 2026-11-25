# -*- coding: utf-8 -*-
"""合成“困难场景”视频帧生成器。

按用户描述的目标特性合成测试素材:
- 目标随机游走移动(带惯性 + 扰动,撞墙反弹);
- 目标形态持续变化(椭圆长短轴、旋转角随时间形变);
- 目标与背景半透明混合(alpha 0.30~0.95 振荡);
- 背景带纹理,且存在多个外观相似的干扰物同步游走。

供 tools/make_test_video.py 生成 test.mp4,以及 tests 与 --selftest
做无 GUI 的自动化验证。
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]


class SyntheticScene:
    """一帧一帧 step() 的合成场景。"""

    TARGET_COLOR = (40, 130, 255)      # 目标主色 (BGR, 橙)
    DISTRACTOR_COLOR = (30, 140, 255)  # 干扰物主色,与目标非常接近

    def __init__(
        self,
        size: Tuple[int, int] = (640, 360),
        seed: int = 7,
        n_distractors: int = 5,
        hide_target_after: Optional[int] = None,
    ) -> None:
        self.w, self.h = size
        self.rng = np.random.default_rng(seed)
        self.hide_target_after = hide_target_after
        self.t = 0

        self.background = self._make_background()

        # 目标状态:位置、速度、形变相位、透明度相位、基础尺寸
        self.pos = np.array([self.w * 0.3, self.h * 0.4])
        self.vel = np.array([2.5, 1.8])
        self.base_axes = (26, 18)
        self.morph_phase = 0.0
        self.alpha_phase = 0.0

        # 干扰物:外观与目标相近,各自游走
        self.distractors = []
        for _ in range(n_distractors):
            self.distractors.append(
                {
                    "pos": np.array(
                        [self.rng.uniform(30, self.w - 30), self.rng.uniform(30, self.h - 30)]
                    ),
                    "vel": self.rng.uniform(-2.2, 2.2, 2),
                    "axes": (
                        self.rng.uniform(16, 30),
                        self.rng.uniform(12, 24),
                    ),
                    "phase": self.rng.uniform(0, 2 * np.pi),
                }
            )

        self.last_target_center: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------ 公开接口

    def step(self) -> Tuple[np.ndarray, Optional[BBox]]:
        """推进一帧,返回 (frame, target_bbox);目标被隐藏时 bbox 为 None。"""
        self._move(self.pos, self.vel, speed_cap=6.0)

        self.morph_phase += 0.13
        self.alpha_phase += 0.075
        self.t += 1

        frame = self.background.copy()
        # 轻微的整体亮度扰动,模拟光照变化
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=float(self.rng.normal(0, 2.0)))

        for d in self.distractors:
            self._move(d["pos"], d["vel"], speed_cap=4.0)
            d["phase"] += 0.1
            self._draw_blob(
                frame,
                d["pos"],
                d["axes"],
                d["phase"],
                alpha=0.55 + 0.2 * np.sin(d["phase"] * 0.7),
                color=self.DISTRACTOR_COLOR,
                with_core=True,
            )

        bbox = None
        hidden = self.hide_target_after is not None and self.t >= self.hide_target_after
        if not hidden:
            axes = self._target_axes()
            # 半透明振荡:α∈[0.40, 0.92],主周期 + 快速小波动叠加,
            # 模拟“目标与背景透明混合”但整体仍可见可辨。
            alpha = float(
                np.clip(
                    0.65 + 0.27 * np.sin(self.alpha_phase)
                    + 0.07 * np.sin(self.alpha_phase * 3.7),
                    0.40, 0.92,
                )
            )
            self._draw_blob(
                frame, self.pos, axes, self.morph_phase * 0.8,
                alpha=alpha, color=self.TARGET_COLOR, with_core=True,
            )
            bbox = self._bbox_from(self.pos, axes, self.morph_phase * 0.8)
            self.last_target_center = (float(self.pos[0]), float(self.pos[1]))
        else:
            self.last_target_center = None

        return frame, bbox

    # ------------------------------------------------------------------ 内部实现

    def _make_background(self) -> np.ndarray:
        """带渐变、随机色块和噪声的纹理背景,避免纯色导致追踪过于简单。"""
        gx, gy = np.meshgrid(np.linspace(0, 1, self.w), np.linspace(0, 1, self.h))
        bg = np.stack(
            [
                40 + 55 * gx,
                30 + 45 * gy,
                70 + 60 * (1 - gx * 0.5),
            ],
            axis=-1,
        ).astype(np.uint8)

        rng = self.rng
        for _ in range(70):  # 随机矩形/圆斑纹理
            x, y = rng.integers(0, self.w), rng.integers(0, self.h)
            w2, h2 = rng.integers(8, 70, 2)
            color = tuple(int(c) for c in rng.integers(20, 200, 3))
            if rng.random() < 0.5:
                cv2.rectangle(bg, (int(x), int(y)), (int(x + w2), int(y + h2)), color, -1)
            else:
                cv2.circle(bg, (int(x), int(y)), int(rng.integers(6, 34)), color, -1)

        noise = rng.integers(0, 14, (self.h, self.w, 1), dtype=np.uint8)
        return cv2.add(bg, np.repeat(noise, 3, axis=2))

    def _move(self, pos: np.ndarray, vel: np.ndarray, speed_cap: float) -> None:
        """带惯性的随机游走,撞墙反弹。"""
        vel += self.rng.normal(0.0, 0.55, 2)
        speed = np.linalg.norm(vel)
        if speed > speed_cap:
            vel *= speed_cap / speed
        vel *= 0.92  # 阻尼,保持运动平滑
        pos += vel
        for i, limit in enumerate((self.w, self.h)):
            if pos[i] < 24:
                pos[i] = 24
                vel[i] = abs(vel[i])
            elif pos[i] > limit - 24:
                pos[i] = limit - 24
                vel[i] = -abs(vel[i])

    def _target_axes(self) -> Tuple[int, int]:
        """目标椭圆长短轴随时间形变。"""
        ax = self.base_axes[0] * (1.0 + 0.38 * np.sin(self.morph_phase))
        ay = self.base_axes[1] * (1.0 - 0.27 * np.sin(self.morph_phase * 1.3))
        return (max(6, int(ax)), max(6, int(ay)))

    @staticmethod
    def _draw_blob(frame, pos, axes, angle_phase, alpha, color, with_core=False) -> None:
        """半透明 + 形变 + 旋转的椭圆斑块(目标和干扰物共用)。"""
        overlay = frame.copy()
        cv2.ellipse(
            overlay,
            (int(pos[0]), int(pos[1])),
            (int(axes[0]), int(axes[1])),
            float(np.degrees(angle_phase)),
            0, 360, color, -1, cv2.LINE_AA,
        )
        if with_core:  # 内部加一点纹理细节,便于相关滤波锁定
            cv2.ellipse(
                overlay,
                (int(pos[0]), int(pos[1])),
                (max(2, int(axes[0] * 0.45)), max(2, int(axes[1] * 0.45))),
                float(np.degrees(angle_phase)) + 45.0,
                0, 360,
                (int(color[0] * 0.5), int(color[1] * 0.5), int(color[2] * 0.5)),
                -1, cv2.LINE_AA,
            )
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)

    def _bbox_from(self, pos, axes, angle_phase) -> BBox:
        """根据椭圆参数与旋转角计算外接正框。"""
        a, b = float(axes[0]), float(axes[1])
        theta = float(angle_phase)
        hw = np.sqrt((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2)
        hh = np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
        x = int(round(pos[0] - hw))
        y = int(round(pos[1] - hh))
        w = int(round(2 * hw))
        h = int(round(2 * hh))
        x = max(0, min(x, self.w - 2))
        y = max(0, min(y, self.h - 2))
        w = max(2, min(w, self.w - x))
        h = max(2, min(h, self.h - y))
        return (x, y, w, h)


class FadeRectScene:
    """「白色矩形渐隐」场景:复刻用户描述的最难情形。

    - 开头约 2 秒:一个白色矩形在画面中央旋转(完全可见,适合框选初始化);
    - 随后 2 秒:矩形逐渐透明,与水纹背景融合,同时开始漂移;
    - 之后:近乎不可见(alpha 0.05~0.15,带轻微闪烁),随机游走;
      唯一可辨的线索是跟随目标一起移动、轻微闪烁的水波纹光晕,
      背景里还烘焙了几组「静态」波纹作为干扰。
    """

    RECT_W, RECT_H = 38, 26          # 矩形尺寸
    FADE_START = 55                  # 开始变透明的帧
    FADE_END = 115                   # 完全融入背景的帧
    RIPPLE_RADII = (34, 50, 66)      # 目标波纹光晕半径

    def __init__(
        self,
        size: Tuple[int, int] = (640, 360),
        seed: int = 3,
        n_static_ripples: int = 3,
    ) -> None:
        self.w, self.h = size
        self.rng = np.random.default_rng(seed)
        self.t = 0

        self.pos = np.array([self.w * 0.5, self.h * 0.5])
        self.vel = np.array([1.2, -0.8])
        self.angle = 0.0

        self.background = self._make_background()
        # 静态干扰波纹:位置固定,只有目标波纹才会动
        self.static_ripples = [
            (self.rng.uniform(60, self.w - 60), self.rng.uniform(50, self.h - 50),
             self.rng.uniform(24, 40))
            for _ in range(n_static_ripples)
        ]

        self.visible = True  # 目标是否仍明显可见(供测试区分阶段)

    # ------------------------------------------------------------------ 公开接口

    def step(self) -> Tuple[np.ndarray, BBox]:
        """推进一帧,返回 (frame, 目标外接框)。"""
        self.t += 1
        t = self.t

        # 透明度曲线:可见 → 线性渐隐 → 近乎不可见并轻微闪烁
        if t <= self.FADE_START:
            alpha = 0.95
            self.visible = True
        elif t <= self.FADE_END:
            k = (t - self.FADE_START) / (self.FADE_END - self.FADE_START)
            alpha = 0.95 - 0.85 * k
            self.visible = alpha > 0.45
        else:
            alpha = 0.10 + 0.05 * (1.0 + np.sin(t * 0.45))  # 0.10 ~ 0.20
            self.visible = False

        # 运动:先慢速漂移,渐隐后随机游走(阻尼 + 扰动,撞墙反弹)
        if t <= self.FADE_START:
            self.vel *= 0.98
        else:
            self.vel += self.rng.normal(0.0, 0.22, 2)
            speed = np.linalg.norm(self.vel)
            cap = 2.8
            if speed > cap:
                self.vel *= cap / speed
            self.vel *= 0.97
        self.pos += self.vel
        for i, limit in enumerate((self.w, self.h)):
            m = 40
            if self.pos[i] < m:
                self.pos[i] = m
                self.vel[i] = abs(self.vel[i])
            elif self.pos[i] > limit - m:
                self.pos[i] = limit - m
                self.vel[i] = -abs(self.vel[i])

        # 旋转:可见阶段稳定旋转,渐隐后放慢
        self.angle += 2.4 if self.visible else 0.9

        frame = self.background.copy()

        # 目标波纹光晕:随目标移动、半径缓慢脉动、亮度闪烁 —— 关键线索
        for k, r0 in enumerate(self.RIPPLE_RADII):
            r = r0 + 4.0 * np.sin(t * 0.5 + k * 1.3)
            a = 0.10 + 0.05 * np.sin(t * 0.9 + k)
            self._blend_ring(frame, self.pos, r, a, color=(215, 235, 245))

        # 静态干扰波纹(烘焙在背景上,不随目标移动)
        for sx, sy, sr in self.static_ripples:
            cv2.circle(frame, (int(sx), int(sy)), int(sr), (205, 225, 235), 1, cv2.LINE_AA)
            cv2.circle(frame, (int(sx), int(sy)), int(sr * 0.7), (200, 220, 232), 1, cv2.LINE_AA)

        # 半透明白色矩形
        self._blend_rect(frame, alpha)

        return frame, self._gt_bbox()

    # ------------------------------------------------------------------ 内部实现

    def _make_background(self) -> np.ndarray:
        """水纹背景:蓝绿渐变 + 干涉波纹 + 深浅斑块 + 噪声(全程静止)。"""
        gx, gy = np.meshgrid(np.linspace(0, 1, self.w), np.linspace(0, 1, self.h))
        xs = np.arange(self.w, dtype=np.float32)
        ys = np.arange(self.h, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        ripple = (
            7.0 * np.sin(xx / 23.0 + yy / 17.0)
            + 5.0 * np.sin(xx / 41.0 - yy / 29.0 + 1.2)
            + 4.0 * np.sin((xx + yy) / 61.0)
        )
        base = np.stack(
            [
                92 + 26 * gx + ripple * 0.9,
                118 + 30 * gy + ripple,
                128 + 24 * (1 - gx * 0.4) + ripple * 0.8,
            ],
            axis=-1,
        ).clip(0, 255).astype(np.uint8)

        for _ in range(26):  # 深浅斑块,模拟水下明暗
            x, y = self.rng.integers(0, self.w), self.rng.integers(0, self.h)
            w2, h2 = self.rng.integers(30, 110, 2)
            shade = int(self.rng.integers(-30, 26))
            roi = base[
                max(0, int(y)) : min(self.h, int(y + h2)),
                max(0, int(x)) : min(self.w, int(x + w2)),
            ]
            if roi.size:
                base[
                    max(0, int(y)) : min(self.h, int(y + h2)),
                    max(0, int(x)) : min(self.w, int(x + w2)),
                ] = np.clip(roi.astype(np.int16) + shade, 0, 255).astype(np.uint8)

        noise = self.rng.integers(-6, 7, (self.h, self.w, 1), dtype=np.int16)
        return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def _blend_rect(self, frame: np.ndarray, alpha: float) -> None:
        if alpha <= 0.01:
            return
        overlay = frame.copy()
        pts = cv2.boxPoints(
            (
                (float(self.pos[0]), float(self.pos[1])),
                (self.RECT_W, self.RECT_H),
                float(np.degrees(self.angle)),
            )
        ).astype(np.int32)
        cv2.fillPoly(overlay, [pts], (245, 248, 252))
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)

    def _blend_ring(self, frame, pos, radius, alpha, color) -> None:
        if alpha <= 0.01:
            return
        overlay = frame.copy()
        cv2.circle(overlay, (int(pos[0]), int(pos[1])), int(radius), color, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)

    def _gt_bbox(self) -> BBox:
        pts = cv2.boxPoints(
            (
                (float(self.pos[0]), float(self.pos[1])),
                (self.RECT_W, self.RECT_H),
                float(np.degrees(self.angle)),
            )
        )
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        x = max(0, int(round(x0)))
        y = max(0, int(round(y0)))
        w = max(2, min(int(round(x1 - x0)), self.w - x))
        h = max(2, min(int(round(y1 - y0)), self.h - y))
        return (x, y, w, h)
