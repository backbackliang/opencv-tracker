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
- **运动能量追踪(最后防线)**:目标与背景完全融合后,唯一线索是它还在动。
  `MotionTracker` 用「多时间基线帧差(1/2/3 帧)+ 衰减残影」累积运动能量,
  「预测门控 + 方向一致性」做数据关联(模拟人眼“按移动方向持续追踪”的先验),
  并带相位相关全局运动补偿与全图重捕获。颜色模型不可靠时(如白色/灰色目标)
  自动以运动追踪为主、CSRT 做局部精修;两者长期背离时把 CSRT 重锚定回目标。

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

# 没有素材?生成内置合成视频(默认“白色矩形渐隐”场景)
python tools/make_test_video.py                 # → test.mp4
python tools/make_test_video.py --scenario blobs # → 彩色斑块+干扰物场景

python run_tracker.py             # 加载当前目录 test.mp4,按 F3 开始框选
```

内置的 `fade-rect` 场景复刻最难情形:白色矩形旋转 2 秒 → 渐隐融入水纹背景 →
近乎不可见地随机移动,唯一线索是随目标移动、轻微闪烁的水波纹光晕
(背景里另有几组静态波纹作干扰),人眼也只能靠移动方向追踪它。

程序默认加载当前目录下的 `test.mp4`;`--tracker kcf` 可强制 KCF,
`--selftest` 运行无 GUI 的端到端自检。

## 项目结构

```
run_tracker.py          # 命令行入口(GUI / 自检)
tracker/
  core.py               # TargetTracker:CSRT/KCF 封装、置信度、丢失判定、snap、重锚定
  motion.py             # MotionTracker:多基线帧差能量 + 方向门控的运动追踪
  synthetic.py          # 合成场景(fade-rect 渐隐矩形 / blobs 彩色斑块)
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

- 若干扰物与目标**颜色、外观几乎相同且同时运动**,运动与外观信号都无法区分:
  此时需人工逐帧确认,发现跟错即按 `F3` 重新框选——这正是逐帧步进交互的设计目的。
- 目标颜色与背景难区分时自动转入「运动追踪为主」模式(控制台会提示),
  但若目标与背景**同时完全静止**,任何算法都无法凭空定位,只能人工重选。
- 渐隐目标的实测精度(压缩视频链路):不可见阶段平均中心误差约 6px,
  随机急转弯时短暂增大,方向先验会自动重新捕获,全程不判丢失。

## 环境要求

- Python 3.9+
- `opencv-contrib-python >= 4.5`(需包含 Tracking 模块;GUI 窗口需要非 headless 版本)
- `numpy >= 1.23`
