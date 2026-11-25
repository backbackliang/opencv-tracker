#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""opencv-tracker 命令行入口。

用法:
    python run_tracker.py                     # 加载当前目录 test.mp4,GUI 追踪
    python run_tracker.py --video your.mp4    # 指定视频
    python run_tracker.py --tracker kcf       # 强制 KCF(默认 CSRT 优先,自动降级)
    python run_tracker.py --selftest          # 无 GUI 的端到端自检

按键(F3 选框 → 回车确认 → 空格逐帧):
    F3 / R    弹出选框窗口:鼠标左键拖出 ROI,回车(ENTER)确认后开始追踪
    SPACE     下一帧并更新追踪(逐帧步进)
    P         自动连续播放 / 暂停
    Q / ESC   退出
"""

from __future__ import annotations

import argparse
import sys

from tracker.app import TrackerApp, VideoOpenError


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_tracker",
        description="OpenCV 交互式目标追踪(CSRT 优先,鼠标框选,空格逐帧)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", "-v", default="test.mp4",
                   help="视频文件路径(默认当前目录 test.mp4)")
    p.add_argument("--tracker", "-t", default="auto",
                   choices=["auto", "csrt", "kcf"],
                   help="底层追踪器:auto=CSRT 优先并自动降级")
    p.add_argument("--fps", type=float, default=24.0,
                   help="自动播放模式的帧率")
    p.add_argument("--no-snap", action="store_true",
                   help="关闭低置信时的模板漂移校正")
    p.add_argument("--selftest", action="store_true",
                   help="不打开 GUI,用合成困难场景做端到端自检")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.selftest:
        from tracker.selftest import run_selftest
        return run_selftest()

    try:
        app = TrackerApp(
            video_path=args.video,
            tracker_type=args.tracker,
            play_fps=args.fps,
            snap=not args.no_snap,
        )
    except VideoOpenError as e:
        print(f"[错误] {e}")
        return 1

    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
