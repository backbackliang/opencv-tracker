# opencv-tracker

基于 OpenCV **CSRT/KCF** 的交互式目标追踪工具。针对「目标随机移动、形态持续变化、
与背景半透明混合、存在大量相似干扰物」的困难场景设计。

## 功能特性

- **追踪算法**:`TargetTracker` 封装 `cv2.TrackerCSRT_create()`(默认,精准度优先,
  支持尺度/旋转自适应),CSRT 不可用时自动降级 `cv2.TrackerKCF_create()`。
- **交互选区**:运行时按 **F3** 弹出选框窗口,鼠标左键拖动绘制 ROI,按 **回车(ENTER)**
  确认后窗口关闭、初始化追踪器开始追踪;按 **R** 可随时重新框选。
- **坐标输出与绘制**:目标框高亮绘制,框上方实时显示中心坐标 `(X, Y)`、置信度
  `conf` 与状态(`TRACKING / LOW_CONF / LOST`);左上角状态栏显示帧号/模式/按键提示。
- **异常重置**:目标丢失时画面中央提示 **TARGET LOST**,自动暂停,按 **F3 / R**
  重新框选即可恢复;视频播完按 **R** 从头再来。
- **逐帧步进**:按 **空格** 逐帧推进(应对随机运动),**P** 切换自动连播。
- **外观置信度**:针对半透明目标专门设计的「特征色占比」置信度(H-S-V 直方图,
  背景环对照 + α 阶梯颜色变体),目标在场约 0.6~1.0、框在背景上趋近 0;
  置信度长期过低即判定丢失(连续 7 帧),低置信时自动做特征色反投影漂移校正(snap)。

## 按键说明

| 按键 | 功能 |
| --- | --- |
| `F3` / `R` | 弹出选框窗口:鼠标左键拖出 ROI,回车(ENTER)确认后开始追踪 |
| `空格` | 下一帧并更新追踪(逐帧步进) |
| `P` | 自动连续播放 / 暂停 |
| `Q` / `ESC` | 退出 |

## 快速开始

```bash
pip install -r requirements.txt   # numpy + opencv-contrib-python

# 用你自己的视频:放到当前目录命名为 test.mp4,或
python run_tracker.py --video 你的视频.mp4

# 没有素材?生成一个内置的合成困难场景视频(test.mp4)
python tools/make_test_video.py

python run_tracker.py             # 加载当前目录 test.mp4,按 F3 开始框选
```

程序默认加载当前目录下的 `test.mp4`;`--tracker kcf` 可强制 KCF,
`--selftest` 运行无 GUI 的端到端自检。

## 项目结构

```
run_tracker.py          # 命令行入口(GUI / 自检)
tracker/
  core.py               # TargetTracker:CSRT/KCF 封装、置信度、丢失判定、snap 校正
  synthetic.py          # 合成困难场景(随机移动/形变/半透明/相似干扰物)
  hud.py                # 追踪框、坐标、置信度、TARGET LOST 等画面叠加绘制
  app.py                # GUI 主循环(F3 选框 / 空格逐帧 / P 连播)
  selftest.py           # 无 GUI 端到端自检
tools/make_test_video.py# 生成合成测试视频 test.mp4
tests/test_core.py      # 单元测试(python -m pytest tests/ 或直接运行)
```

## 测试

```bash
python -m pytest tests/ -q      # 单元测试
python run_tracker.py --selftest# 端到端自检(含追踪精度/丢失判定断言)
```

## 已知边界(外观方法的本质限制)

- 若干扰物与目标**颜色、外观几乎相同**,外观信号无法区分:置信度会保持高位,
  此时需人工逐帧确认,发现跟错即按 `F3` 重新框选——这正是逐帧步进交互的设计目的。
- 目标整体不比背景鲜艳(如灰暗目标配彩色背景)时,颜色模型自动退化为
  「仅依赖 tracker 返回值」模式(启动时控制台会提示)。

## 环境要求

- Python 3.9+
- `opencv-contrib-python >= 4.5`(需包含 Tracking 模块;GUI 窗口需要非 headless 版本)
- `numpy >= 1.23`
