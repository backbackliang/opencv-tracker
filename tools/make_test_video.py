#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成合成“困难场景”测试视频(默认输出当前目录 test.mp4)。

场景特性与真实难题对应:
- 目标随机游走移动(惯性 + 扰动);
- 形态持续变化(椭圆长短轴/旋转角形变);
- 与背景半透明混合(alpha 0.30~0.95 振荡);
- 纹理背景 + 多个外观相似的干扰物同步游走。

用法:
    python tools/make_test_video.py                    # test.mp4, 240帧, 640x360
    python tools/make_test_video.py -o demo.mp4 -n 300
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from tracker.synthetic import SyntheticScene


def main() -> int:
    p = argparse.ArgumentParser(description="生成合成追踪测试视频")
    p.add_argument("-o", "--output", default="test.mp4", help="输出文件名")
    p.add_argument("-n", "--frames", type=int, default=240, help="帧数")
    p.add_argument("--fps", type=int, default=30, help="帧率")
    p.add_argument("--size", default="640x360", help="分辨率,如 640x360")
    p.add_argument("--seed", type=int, default=7, help="随机种子")
    args = p.parse_args()

    try:
        w, h = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        print(f"[错误] --size 格式应为 WxH,收到 {args.size!r}")
        return 1

    scene = SyntheticScene(size=(w, h), seed=args.seed, n_distractors=5)

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))
    if not writer.isOpened():  # 个别环境缺 mp4v 编码器时回退 AVI
        alt = os.path.splitext(args.output)[0] + ".avi"
        print(f"[警告] mp4v 编码器不可用,改写 {alt}")
        writer = cv2.VideoWriter(alt, cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (w, h))
        if not writer.isOpened():
            print("[错误] 无法创建视频写出器")
            return 1
        args.output = alt

    for _ in range(args.frames):
        frame, _ = scene.step()
        writer.write(frame)
    writer.release()

    size_kb = os.path.getsize(args.output) / 1024
    print(f"已生成 {args.output}({args.frames} 帧, {w}x{h}, {size_kb:.0f} KB)")
    print("运行 python run_tracker.py 即可加载它进行交互式追踪。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
