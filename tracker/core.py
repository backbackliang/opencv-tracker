# -*- coding: utf-8 -*-
"""目标追踪核心模块。

封装 CSRT(优先,精准度更高、支持尺度/旋转自适应)与 KCF(兜底)两种
OpenCV 原生追踪器,并在此基础上补充三件原生 tracker 不提供的能力:

1. 外观置信度(目标特征色占比):
   半透明目标的 ROI 里混有大量背景像素,直接拿 ROI 做模板匹配会被背景
   纹理主导(背景自相关可达 0.3~0.5),导致「框在背景上」也报高置信。
   因此初始化时用「ROI 周围背景环的饱和度分布」做参照,只把明显比背景
   更饱和的像素建模为“目标特征色”(3D HSV 直方图,平滑后归一化);
   之后每帧统计当前框内命中特征色的像素占比,并按初始占比归一,得到
   0~1 的置信度。目标在场时约 0.6~1.0,框在背景上时接近 0,区分度高。

2. 丢失判定:
   - tracker 返回失败 / 框越界 → 立即丢失;
   - 置信度连续 lost_streak 帧低于 conf_lost → 判定 LOST
     (抗半透明振荡与短暂遮挡,不会一帧低就误报)。

3. 漂移校正(snap):
   置信度偏低时,在当前框附近窗口内对「特征色概率图」做反投影 + 高斯
   平滑,若峰值足够强则把框中心拉回峰值处——相似背景纹理不会形成峰值,
   只有真正像目标颜色的区域才会,因此比灰度模板匹配可靠得多。

已知边界(设计取舍):
   - 若目标外观与背景/干扰物颜色几乎相同,任何外观信号都无法区分,
     置信度会保持高位;此时由用户在逐帧查看中发现并按 F3 重选。
   - 目标整体低于背景饱和度(如灰色目标配彩色背景)时,特征色不可靠,
     模块会自动退化为「只依赖 tracker 自身返回值」的模式。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .motion import MotionTracker

BBox = Tuple[int, int, int, int]

# 不同 OpenCV 版本/构建中 CSRT/KCF 的注册位置不同,逐个尝试。
_CSRT_FACTORIES = (
    "TrackerCSRT_create",            # 常规 opencv-contrib-python 4.x/5.x
    "legacy.TrackerCSRT_create",     # 部分 4.5.x 构建的旧命名空间
    "TrackerCSRT.create",            # 新式静态工厂
)
_KCF_FACTORIES = (
    "TrackerKCF_create",
    "legacy.TrackerKCF_create",
    "TrackerKCF.create",
)

_HIST_BINS = 8   # H/S 通道的 bin 数
_V_BINS = 8      # V(亮度)bin 数;配合 α 阶梯变体覆盖亮度摆动


class TrackStatus(enum.Enum):
    """追踪状态,绘制在目标框上方。"""

    IDLE = "IDLE"            # 尚未选定目标
    TRACKING = "TRACKING"    # 正常追踪
    LOW_CONF = "LOW_CONF"    # 置信度偏低(形变/遮挡/疑似漂移),建议人工确认
    LOST = "LOST"            # 目标丢失,等待重新框选


@dataclass
class TrackResult:
    """一帧的追踪结果。"""

    status: TrackStatus
    bbox: Optional[BBox] = None                      # 当前(或最后有效)边界框
    center: Optional[Tuple[int, int]] = None         # 目标中心 (X, Y)
    confidence: float = 0.0                          # 外观置信度 0~1(相对初始)
    note: str = ""                                   # 附加信息,如 "snap"


def _sanitize_bbox(bbox, frame_shape) -> Optional[BBox]:
    """把任意 bbox 规整为画布内的合法整数 bbox;无效时返回 None。"""
    if bbox is None:
        return None
    H, W = frame_shape[:2]
    x, y, w, h = (int(round(float(v))) for v in bbox[:4])
    if w <= 0 or h <= 0:
        return None
    x = min(max(x, 0), W - 2)
    y = min(max(y, 0), H - 2)
    w = min(w, W - x)
    h = min(h, H - y)
    if w <= 1 or h <= 1:
        return None
    return (x, y, w, h)


def _resolve_factory(paths):
    """按路径逐个查找 tracker 工厂函数,找不到返回 None。"""
    for path in paths:
        obj = cv2
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if callable(obj):
            return obj
    return None


def create_tracker(tracker_type: str = "CSRT"):
    """创建底层追踪器。

    优先 CSRT(精准度优先,支持尺度与旋转自适应);CSRT 不可用时自动
    降级 KCF(更快但无尺度自适应)。返回 (tracker 实例, 实际类型名)。
    """
    requested = (tracker_type or "CSRT").upper()
    t = "CSRT" if requested in ("AUTO", "CSRT", "") else requested

    if t == "CSRT":
        factory = _resolve_factory(_CSRT_FACTORIES)
        if factory is not None:
            return factory(), "CSRT"
        t = "KCF"  # CSRT 缺失,按需求降级 KCF

    if t == "KCF":
        factory = _resolve_factory(_KCF_FACTORIES)
        if factory is not None:
            return factory(), "KCF"

    raise RuntimeError(
        "当前 OpenCV 不包含 Tracking 模块,请安装完整版:\n"
        "  pip install opencv-contrib-python>=4.5"
    )


def _hs_bins(img: np.ndarray):
    """返回图像的 (H, S, V) 特征 bin 索引。

    bin 划分针对半透明目标设计:
    - H、S 各 8 bin(0-31、饱和度每档 32);
    - V 只分 4 bin(每档 64):半透明目标的表观亮度 V ≈ α·255 + (1-α)·V_背景
      摆动极大但只跨 2 个粗 bin,而暗背景稳定在更低的 bin,粗分恰好
      吸收亮度摆动、保留「目标比背景亮」这一判别信息。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    b = _HIST_BINS
    return (
        np.clip((hsv[:, :, 0] / 32).astype(np.intp), 0, b - 1),
        np.clip((hsv[:, :, 1] / 32).astype(np.intp), 0, b - 1),
        np.clip((hsv[:, :, 2] / 64).astype(np.intp), 0, _V_BINS - 1),
    )


def _smooth_sig(hist: np.ndarray) -> np.ndarray:
    """特征直方图平滑;H 轴按圆形处理(0° 与 360° 相邻)。"""
    out = hist
    for axis in range(3):
        plus = np.roll(out, 1, axis)
        minus = np.roll(out, -1, axis)
        out = (plus + 2.0 * out + minus) / 4.0
    return out


class TargetTracker:
    """封装 CSRT/KCF 的目标追踪器。

    参数:
        tracker_type: "CSRT"(默认/优先)、"KCF" 或 "AUTO"。
        conf_warn:    置信度低于该值显示 LOW_CONF 预警并触发漂移校正。
        conf_lost:    置信度低于该值计入「疑似丢失」连击计数。
        lost_streak:  连续多少帧疑似丢失后判定 LOST。
        snap:         是否启用低置信时的特征色漂移校正。
        snap_min:     反投影峰值达到该值才把框拉回目标。
        snap_search:  漂移校正的搜索半径倍数(相对当前框尺寸)。
        motion:       是否启用运动能量追踪(目标与背景融合后的最后防线:
                      帧差能量 + 方向门控,模拟人眼“按移动方向持续追踪”)。
        motion_min:   运动证据的最低能量强度(相对背景噪声水平)。
    """

    SIG_BAND = 20       # 初始化时提取背景环的宽度(像素)
    SIG_THRESHOLD = 0.40  # 特征色命中阈值(概率图归一化后)

    def __init__(
        self,
        tracker_type: str = "CSRT",
        conf_warn: float = 0.55,
        conf_lost: float = 0.50,
        lost_streak: int = 7,
        snap: bool = True,
        snap_min: float = 0.50,
        snap_search: float = 1.6,
        motion: bool = True,
        motion_min: float = 1.6,
    ) -> None:
        self.tracker_type = tracker_type
        self.conf_warn = float(conf_warn)
        self.conf_lost = float(conf_lost)
        self.lost_streak = int(lost_streak)
        self.snap_enabled = bool(snap)
        self.snap_min = float(snap_min)
        self.snap_search = float(snap_search)
        self.motion_enabled = bool(motion)
        self.motion_min = float(motion_min)

        self._tracker = None
        self._resolved_name: str = ""
        self._signature: Optional[np.ndarray] = None  # 目标特征色概率图 8^3
        self._sig_scale: float = 0.0                  # 初始帧的特征色占比
        self._sig_reliable: bool = False              # 特征色是否足以驱动丢失判定
        self._miss = 0                                # 疑似丢失连击计数
        self._motion = MotionTracker()
        self._csrt_diverge = 0                        # CSRT 框与修正框持续背离帧数
        self._csrt_fail = 0                           # CSRT 连续失败帧数
        self._motion_miss = 0                         # 连续无运动证据帧数
        self.last_result = TrackResult(TrackStatus.IDLE)

    # ------------------------------------------------------------------ 属性

    @property
    def active(self) -> bool:
        """是否已选定目标并初始化追踪。"""
        return self._tracker is not None

    @property
    def tracker_name(self) -> str:
        return self._resolved_name or "-"

    # ------------------------------------------------------------------ 初始化

    def init(self, frame: np.ndarray, bbox) -> bool:
        """用鼠标框选的 ROI 初始化(或重新初始化)追踪器。"""
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("init() 收到空帧")
        clean = _sanitize_bbox(bbox, frame.shape)
        if clean is None:
            return False

        tracker, resolved = create_tracker(self.tracker_type)
        tracker.init(frame, clean)

        self._build_signature(frame, clean)
        self._tracker = tracker
        self._resolved_name = resolved
        self._miss = 0
        self._csrt_diverge = 0
        self._csrt_fail = 0
        self._motion.reset()
        self._motion_miss = 0
        cx, cy = self._center_of(clean)
        self._motion.last_center = np.array([float(cx), float(cy)])
        self.last_result = TrackResult(
            TrackStatus.TRACKING, clean, self._center_of(clean), 1.0, note="init"
        )
        return True

    def reset(self) -> None:
        """完全复位,回到未选目标状态。"""
        self._tracker = None
        self._resolved_name = ""
        self._signature = None
        self._sig_scale = 0.0
        self._sig_reliable = False
        self._miss = 0
        self._motion.reset()
        self._motion_miss = 0
        self._csrt_diverge = 0
        self._csrt_fail = 0
        self.last_result = TrackResult(TrackStatus.IDLE)

    # ------------------------------------------------------------------ 逐帧更新

    def update(self, frame: np.ndarray) -> TrackResult:
        """推进一帧:返回追踪状态、边界框、中心坐标与置信度。

        低置信时的处理顺序:
        1. 特征色反投影 snap(外观还剩一点信息时,把框拉回目标);
        2. 运动能量定位(外观完全失效时,按“预测位置 + 移动方向”跟着走);
        3. CSRT 持续背离/失败时,用当前估计位置重建 CSRT(保留颜色模型)。
        """
        if self._tracker is None:
            return TrackResult(TrackStatus.IDLE)
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("update() 收到空帧")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        csrt_ok = True
        try:
            res = self._tracker.update(frame)
        except cv2.error:
            res = (False, None)

        # 兼容不同版本:update() 可能返回 (ok, bbox) 元组,也可能只返回 bbox。
        raw_bbox = None
        if isinstance(res, (tuple, list)) and len(res) >= 2:
            csrt_ok, raw_bbox = bool(res[0]), res[1]
        elif res is not None:
            raw_bbox = res

        clean = _sanitize_bbox(raw_bbox, frame.shape)
        if clean is None:
            # CSRT 失败或框非法:退回上一有效框继续运动追踪,而不是立刻判丢失。
            csrt_ok = False
            clean = self.last_result.bbox
            if clean is None:
                return self._mark_lost("tracker 失败且无历史框")
            clean = _sanitize_bbox(clean, frame.shape)

        conf = self._signature_conf(frame, clean)
        note = ""

        # 1) 特征色反投影漂移校正。
        if self.snap_enabled and self._sig_reliable and conf < self.conf_warn:
            deep = conf < self.conf_lost
            snapped = self._try_snap(
                frame, clean, search_mult=self.snap_search,
                max_shift_mult=1.6 if deep else 0.9,
            )
            if snapped is not None:
                clean = snapped
                conf = self._signature_conf(frame, clean)
                note = "snap"

        # 2) 运动能量定位:外观失效时的最后防线。
        #    颜色模型不可靠(如白色/灰色目标)时,运动定位每帧都运行,
        #    作为 CSRT 的独立验证器:一旦两者长期分歧就纠正并重锚定。
        #    连续多帧无运动证据时扩大到全图重捕获(要求更强证据)。
        motion_hit = False
        motion_active = (
            conf < self.conf_warn
            or not csrt_ok
            or not self._sig_reliable
        )
        if self.motion_enabled and motion_active:
            reacquire = self._motion_miss >= 5
            pred = None
            gate = 40.0
            if not reacquire and self._motion.last_center is not None:
                speed = float(np.linalg.norm(self._motion.vel))
                pred = (
                    self._motion.last_center[0] + self._motion.vel[0],
                    self._motion.last_center[1] + self._motion.vel[1],
                )
                gate = max(28.0, 2.5 * speed + 0.9 * max(clean[2], clean[3]))
            size_hint = max(clean[2], clean[3]) / 2.0
            need = self.motion_min * 1.7 if reacquire else self.motion_min
            loc, strength = self._motion.locate(gray, pred, gate, size_hint)
            if loc is not None and strength >= need:
                x, y, w, h = clean
                cx_cur, cy_cur = x + w / 2.0, y + h / 2.0
                # 偏差明显时才纠正,避免与 CSRT 正常结果互相微扰。
                far = np.hypot(loc[0] - cx_cur, loc[1] - cy_cur) > 0.45 * max(w, h)
                if not csrt_ok or far:
                    shifted = _sanitize_bbox(
                        (loc[0] - w / 2, loc[1] - h / 2, w, h), frame.shape
                    )
                    if shifted is not None:
                        clean = shifted
                        motion_hit = True
                        motion_conf = min(0.9, 0.35 + 0.15 * strength)
                        if self._sig_reliable:
                            conf = max(conf, motion_conf)
                        note = "motion" if not note else note + "+motion"
                        # 颜色模型不可靠时运动是主依据:立即把 CSRT 锚定到
                        # 目标当前位置,防止被拒帧回退到 CSRT 的旧位置。
                        if not self._sig_reliable:
                            self._re_anchor(frame, clean)
                else:
                    motion_hit = True  # 运动证据与 CSRT 一致:目标在场
                self._motion_miss = 0
            else:
                self._motion_miss += 1

        # 3) CSRT 与最终估计长期背离 / 连续失败时重建 CSRT(保留颜色模型)。
        if csrt_ok:
            csrt_c = self._center_of(_sanitize_bbox(raw_bbox, frame.shape) or clean)
            if np.linalg.norm(np.subtract(csrt_c, self._center_of(clean))) > 1.3 * max(
                clean[2], clean[3]
            ):
                self._csrt_diverge += 1
            else:
                self._csrt_diverge = 0
            self._csrt_fail = 0
        else:
            self._csrt_fail += 1
        if (self._csrt_diverge >= 8 or self._csrt_fail >= 5) and self.last_result.bbox is not None:
            self._re_anchor(frame, clean)
            self._csrt_diverge = 0
            self._csrt_fail = 0

        # 丢失判定:无外观证据且无运动证据,连续多帧 → 判定 LOST。
        # 颜色模型不可靠时,以「CSRT 正常或存在运动证据」作为在场依据。
        if self._sig_reliable:
            present = motion_hit or conf >= self.conf_lost
        else:
            present = motion_hit or csrt_ok
        if present:
            self._miss = 0
        else:
            self._miss += 1

        if self._miss >= self.lost_streak:
            return self._mark_lost(f"置信度连续偏低({self._miss}帧)")
        if conf < self.conf_warn:
            status = TrackStatus.LOW_CONF
        else:
            status = TrackStatus.TRACKING

        result = TrackResult(status, clean, self._center_of(clean), conf, note)
        self.last_result = result
        return result

    def _re_anchor(self, frame: np.ndarray, bbox: BBox) -> None:
        """用当前估计位置重建底层 CSRT(保留特征色模型,不清空外观记忆)。"""
        tracker, resolved = create_tracker(self.tracker_type)
        tracker.init(frame, bbox)
        self._tracker = tracker
        self._resolved_name = resolved
        print(f"[重锚定] tracker 已在 {bbox} 处重建({resolved}),颜色模型保持不变")

    # ------------------------------------------------------------------ 特征色模型

    def _build_signature(self, frame: np.ndarray, bbox: BBox) -> None:
        """初始化目标特征色概率图(2D H-S 模型)。

        半透明目标的表观颜色随透明度振荡:亮度 V ≈ α·255+(1-α)·V_背景
        摆动极大,色相 H 基本不变,饱和度 S 温和下降。因此这里只用
        (H, S) 两维建模,并沿 α 阶梯合成“变透明后”的颜色变体,覆盖整个
        振荡区间。权重只给「明显比背景环更饱和」的像素,抑制背景色混入。

        若目标颜色并不比背景突出(高饱和像素过少),标记为不可靠,
        退化为基础追踪模式。
        """
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        band = self.SIG_BAND
        x0, y0 = max(0, x - band), max(0, y - band)
        x1, y1 = min(W, x + w + band), min(H, y + h + band)
        ring = frame[y0:y1, x0:x1]
        ring_mask = np.ones(ring.shape[:2], bool)
        ring_mask[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = False

        patch_hsv = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
        ring_hsv = cv2.cvtColor(ring, cv2.COLOR_BGR2HSV).astype(np.float32)
        s_bg = float(np.percentile(ring_hsv[:, :, 1][ring_mask], 75)) / 255.0

        sat = patch_hsv[:, :, 1].astype(np.float32) / 255.0
        weight = np.clip((sat - s_bg) / 0.25, 0.0, None) ** 2

        mask = weight > 0
        hist = np.zeros((_HIST_BINS, _HIST_BINS, _V_BINS), np.float32)
        if int(mask.sum()) >= 20:
            # 目标像素颜色集合,沿 α 阶梯生成“变透明”变体
            colors_bgr = frame[y : y + h, x : x + w][mask].astype(np.float32)
            weights = weight[mask].astype(np.float32)
            bg_bgr = ring[ring_mask].mean(axis=0)  # 背景平均色,作为混色基准
            for alpha in (1.0, 0.7, 0.5):
                blended = alpha * colors_bgr + (1.0 - alpha) * bg_bgr
                blended = np.clip(blended, 0, 255).astype(np.uint8).reshape(1, -1, 3)
                hsv = cv2.cvtColor(blended, cv2.COLOR_BGR2HSV)[0].astype(np.float32)
                hb = np.clip(hsv[:, 0] / 32, 0, _HIST_BINS - 1).astype(np.intp)
                sb = np.clip(hsv[:, 1] / 32, 0, _HIST_BINS - 1).astype(np.intp)
                vb = np.clip(hsv[:, 2] / 64, 0, _V_BINS - 1).astype(np.intp)
                np.add.at(hist, (hb, sb, vb), weights)
            hist = _smooth_sig(hist)
            peak = float(hist.max())

        self._signature = hist / peak if peak > 0 else None

        self._sig_scale = self._raw_signature_score(frame, bbox)
        # 判别力检验:签名在「背景环」上的命中率应明显低于 ROI。
        # 若背景环几乎同样命中(如白色目标,签名其实来自背景色),
        # 说明签名只是背景噪声,标记为不可靠。
        band_x, band_y = x0, y0
        band_w, band_h = x1 - x0, y1 - y0
        band_score = 0.0
        if self._signature is not None:
            band_bbox = _sanitize_bbox((band_x, band_y, band_w, band_h), frame.shape)
            if band_bbox is not None:
                band_score = self._raw_signature_score(frame, band_bbox)
        discriminative = (
            self._sig_scale <= 0 or band_score / max(self._sig_scale, 1e-3) <= 0.6
        )
        # 特征色占比过低 → 颜色信号不可靠(目标不比背景饱和/框内全是背景色)
        self._sig_reliable = (
            self._sig_scale >= 0.08 and self._signature is not None and discriminative
        )
        if not self._sig_reliable:
            print(
                "[提示] 目标颜色与背景难以区分,置信度仅作参考;"
                "将启用运动能量追踪作为主要依据,请留意画面人工确认。"
            )

    def _raw_signature_score(self, frame: np.ndarray, bbox: BBox) -> float:
        """当前框内「特征色像素占比」(中心加权,未归一)。"""
        if self._signature is None:
            return 0.0
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, W - x), min(h, H - y)
        if w <= 1 or h <= 1:
            return 0.0
        patch = frame[y : y + h, x : x + w]
        hb, sb, vb = _hs_bins(patch)
        prob = self._signature[hb, sb, vb]
        hit = (prob > self.SIG_THRESHOLD).astype(np.float32)
        gy, gx = np.mgrid[0:h, 0:w]
        cw = np.exp(-(((gy - h / 2) / (h / 1.8)) ** 2 + ((gx - w / 2) / (w / 1.8)) ** 2))
        denom = max(float(np.mean(cw)), 1e-6)
        return float(np.mean(hit * cw)) / denom

    def _signature_conf(self, frame: np.ndarray, bbox: BBox) -> float:
        """置信度 = 特征色占比 / 初始占比,截断到 [0,1]。"""
        if not self._sig_reliable or self._sig_scale <= 0:
            return 1.0  # 颜色信号不可靠时显示满置信,仅依赖 tracker 返回值
        return max(0.0, min(1.0, self._raw_signature_score(frame, bbox) / self._sig_scale))

    # ------------------------------------------------------------------ 漂移校正

    def _try_snap(
        self,
        frame: np.ndarray,
        bbox: BBox,
        search_mult: float = 1.6,
        max_shift_mult: float = 0.9,
    ) -> Optional[BBox]:
        """特征色反投影漂移校正:在当前框附近窗口内找特征色峰值。

        半透明/形变目标的概率图里常有多个等强峰(相似干扰物、背景色块),
        直接取全局最大值会在它们之间来回横跳。因此这里在所有足够强的峰中
        选择「离当前框中心最近」的一个,并限制单次校正距离,只做小步拉回。
        """
        if self._signature is None:
            return None
        H, W = frame.shape[:2]
        x, y, w, h = bbox
        mx = max(int(w * search_mult), 8)
        my = max(int(h * search_mult), 8)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(W, x + w + mx), min(H, y + h + my)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None

        roi = frame[y0:y1, x0:x1]
        hb, sb, vb = _hs_bins(roi)
        prob = self._signature[hb, sb, vb].astype(np.float32)
        prob = cv2.GaussianBlur(prob, (0, 0), 3)
        _, peak, _, _ = cv2.minMaxLoc(prob)
        if peak < self.snap_min:
            return None

        # 收集所有 >= max(0.85*peak, snap_min) 的候选峰,取离当前中心最近者。
        strong = prob >= max(0.85 * peak, self.snap_min)
        ys, xs = np.nonzero(strong)
        if len(xs) == 0:
            return None
        cx_cur, cy_cur = x + w / 2.0, y + h / 2.0
        dist = (xs + x0 - cx_cur) ** 2 + (ys + y0 - cy_cur) ** 2
        # 位移上限:轻度低置信只许小步拉回,防止跳到相似干扰物上。
        max_d2 = (max_shift_mult * max(w, h)) ** 2
        i_min = int(np.argmin(dist))
        if dist[i_min] > max_d2:
            return None

        cx_new = float(xs[i_min]) + x0
        cy_new = float(ys[i_min]) + y0
        nx = int(round(cx_new - w / 2))
        ny = int(round(cy_new - h / 2))
        return _sanitize_bbox((nx, ny, w, h), frame.shape)

    # ------------------------------------------------------------------ 内部工具

    @staticmethod
    def _center_of(bbox: BBox) -> Tuple[int, int]:
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def _mark_lost(self, reason: str) -> TrackResult:
        self._miss += 1
        result = TrackResult(
            TrackStatus.LOST, self.last_result.bbox, self.last_result.center, 0.0, reason
        )
        self.last_result = result
        return result
