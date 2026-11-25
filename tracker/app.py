# -*- coding: utf-8 -*-
"""交互式目标追踪主程序(GUI 主循环)。

交互流程(与需求一致):
    F3            弹出选框窗口,按住鼠标左键拖出 ROI;
    ENTER(回车)  确认 ROI,关闭选框窗口,初始化追踪器开始追踪;
    SPACE(空格)  读取下一帧并更新追踪(逐帧步进,应对随机运动/形变);
    P             自动连续播放 / 暂停切换;
    F3 或 R       任意时刻重新框选(丢失或漂移后的重置手段);
    Q / ESC       退出。

目标丢失时画面中央提示 "TARGET LOST",并自动暂停等待重新框选。
"""

from __future__ import annotations

import time
from typing import Optional

import cv2

from .core import TrackStatus, TargetTracker
from .hud import draw_frame_overlay

# F3 在不同平台后端下的键值:Windows VK=114,GTK=65472,Qt=16777268。
# 'r' 也将作为重选快捷键(在 Windows 上 F3 原始键值恰好为 114,行为一致)。
_KEY_SELECT = {114, 65472, 16777268}
_KEY_QUIT = {27, 16777217}  # ESC(GTK/Windows 与 Qt)

MAIN_WIN = "OpenCV Target Tracker"
SELECT_WIN = "Select Target - drag left mouse, ENTER to confirm"


class VideoOpenError(RuntimeError):
    """视频无法打开。"""


class TrackerApp:
    def __init__(
        self,
        video_path: str,
        tracker_type: str = "auto",
        play_fps: float = 24.0,
        snap: bool = True,
    ) -> None:
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise VideoOpenError(
                f"无法打开视频文件: {video_path!r}\n"
                "提示:可将你的视频放到当前目录并命名为 test.mp4,\n"
                "或运行 python tools/make_test_video.py 生成一个合成测试视频。"
            )
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise VideoOpenError(f"视频 {video_path!r} 打开了但读不出任何帧。")

        self.frame = frame
        self.frame_idx = 0
        self.tracker = TargetTracker(tracker_type=tracker_type, snap=snap)
        self.playing = False
        self.ended = False
        self.play_fps = max(1.0, float(play_fps))

    # ------------------------------------------------------------------ 主循环

    def run(self) -> None:
        print(__doc__.strip().split("\n\n")[0])
        print("-" * 64)
        print("按键说明:")
        print("  F3 / R    弹出选框窗口:按住鼠标左键拖出目标 ROI,回车(ENTER)确认")
        print("  SPACE     下一帧并更新追踪(逐帧步进)")
        print("  P         自动连续播放 / 暂停")
        print("  Q / ESC   退出")
        print("-" * 64)

        cv2.namedWindow(MAIN_WIN, cv2.WINDOW_NORMAL)

        while True:
            if self.playing:
                self._advance()

            self._render()
            delay = 0 if not self.playing else max(1, int(1000 / self.play_fps))
            key = cv2.waitKey(delay)
            if key == -1:  # 个别后端 waitKey(0) 不阻塞时避免热循环
                time.sleep(0.02)
                continue

            base = key & 0xFF

            if key in _KEY_QUIT or base == ord("q"):
                print("退出。")
                break
            elif key in _KEY_SELECT or base == ord("r"):
                if self.ended:
                    self._restart()
                self._select_target()
            elif base == 32:  # SPACE:下一帧
                if self.ended:
                    print("视频已播完:按 R 从第 0 帧重新开始并重新框选目标。")
                    continue
                self._advance()
            elif base in (ord("p"), ord("P")):
                if not self.tracker.active:
                    print("尚未选定目标:先按 F3/R 框选。")
                    continue
                if self.tracker.last_result.status == TrackStatus.LOST or self.ended:
                    print("目标丢失/视频结束,自动播放已禁用,请先按 F3/R 重新框选。")
                    continue
                self.playing = not self.playing
                print("自动播放:开启" if self.playing else "自动播放:暂停")

        self.cap.release()
        cv2.destroyWindow(MAIN_WIN)

    # ------------------------------------------------------------------ 交互动作

    def _select_target(self) -> None:
        """弹出选框窗口:左键拖出 ROI,回车/空格确认,选框窗口随后关闭。"""
        self.playing = False
        bbox = cv2.selectROI(SELECT_WIN, self.frame, showCrosshair=True, fromCenter=False)
        try:
            cv2.destroyWindow(SELECT_WIN)
        except cv2.error:
            pass

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            print("已取消框选(空 ROI)。")
            return

        ok = self.tracker.init(self.frame, (int(x), int(y), int(w), int(h)))
        if ok:
            print(
                f"[初始化] tracker={self.tracker.tracker_name} "
                f"bbox=({x},{y},{w},{h}) frame={self.frame_idx}"
            )
            print("开始追踪:按 SPACE 逐帧推进,P 切换自动播放。")
        else:
            print("ROI 无效(超出画面或太小),请重新按 F3 框选。")

    def _advance(self) -> None:
        """读取下一帧并更新追踪器;目标丢失时自动暂停。"""
        if self.ended:
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.ended = True
            self.playing = False
            print("视频已到末尾。按 R 重头开始,或 Q 退出。")
            return

        self.frame = frame
        self.frame_idx += 1

        if not self.tracker.active:
            return

        result = self.tracker.update(frame)
        self._log_result(result)
        if result.status == TrackStatus.LOST:
            self.playing = False
            print("!! 目标丢失:画面提示 TARGET LOST,按 F3 或 R 重新框选。")

    def _restart(self) -> None:
        self.tracker.reset()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.cap.read()
        if ok:
            self.frame = frame
            self.frame_idx = 0
        self.ended = False
        print("已回到第 0 帧,请重新框选目标。")

    # ------------------------------------------------------------------ 绘制与日志

    def _render(self) -> None:
        draw_frame_overlay(
            self.frame,
            self.frame_idx,
            self.tracker.tracker_name,
            self.tracker.last_result,
            self.playing,
            self.ended,
        )
        cv2.imshow(MAIN_WIN, self.frame)

    def _log_result(self, result) -> None:
        if result.bbox is None or result.center is None:
            print(f"[frame {self.frame_idx:04d}] status={result.status.value} conf=0.00")
            return
        x, y, w, h = result.bbox
        cx, cy = result.center
        tag = " [snap]" if result.note == "snap" else ""
        print(
            f"[frame {self.frame_idx:04d}] center=({cx},{cy}) "
            f"bbox=({x},{y},{w},{h}) conf={result.confidence:.2f} "
            f"status={result.status.value} tracker={self.tracker.tracker_name}{tag}"
        )
