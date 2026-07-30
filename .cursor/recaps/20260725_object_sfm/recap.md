# 会话复盘：FoundationPose → object_6d_pose_annotation

## 背景目标

- 原工作区：`/root/FoundationPose`（uv 跑通官方 demo）。
- 目标：手机绕拍静物、**无 ARKit**，经 SfM + 米制尺度 + 浏览器 6D 标注 → YOLO6D（含全视频）。
- 当前工程：`/root/ultralytics/projects/object_6d_pose_annotation`（由 `object_sfm` 重命名）。

## 关键决策（全程）

1. 标本段绕拍：用 **SfM 位姿 + 已知尺寸**，不必 BundleSDF→FoundationPose（推理要 depth）。
2. 匹配：**hloc = SuperPoint + LightGlue + COLMAP**（对齐 OnePose 系深度特征思路）。
3. 默认重建配置取 **run1**（1600 / 较快）；run2 原生 4K 太慢且结构收益有限。
4. 稠密：CUDA COLMAP MVS（run1）可用；Depth-Anything 融合几何差，不用作主路径；OpenMVS 预编译纳入 `third_party/setup.sh`。
5. 全视频标签：对每帧相对 SfM 做 **hloc 定位**（非插值），再投影静态 `object_frame`。
6. 物体框约定：**+Z ∥ 桌面法向**，**−Z 面贴桌**；导出优先用 `quaternion_wxyz`。
7. 仓库：大产物 gitignore；`.cursor` 入库；环境用 **uv sync**（`pyproject.toml` + `uv.lock`）。

## 阶段 A — FoundationPose（已完成）

- 路径：`/root/FoundationPose`，GPU：3080 Ti 12GB。
- `run_demo.py` 无头保存可视化；文档见该仓库 `docs/`。

## 阶段 B — SfM / 稠密（已完成）

| 项 | 内容 |
|----|------|
| 视频 | `data/VID_20260725_165829.mp4`，3840×2160，~1097 帧 |
| 抽帧 | `data/frames/` 74 张 1600×900 @~2fps；`frames_native/` 同采样原生分辨率 |
| SfM | `outputs/run1/sfm`，**74/74**，~8447 点 |
| 稠密 | `outputs/run1/export/dense_mvs.ply`（CUDA COLMAP）；标注场景包用其降采样点云 |

## 阶段 C — 浏览器标注器（已完成）

路径：`tools/pose_annotator/`（`server.py`、`static/app.js`、`export_bridge.py`、`scale_bridge.py`、`segment.py`）。

已实现要点：

- 点云查看：轨道旋转、点大小、旋转中心（STR 等）。
- 多视图 2D + **SAM2** 交互分割 → 前景点云交集 → AABB/OBB 初值 → TransformControls 精修。
- 尺度：选点拟合平面 → 法向 offset → 选目标点 + 真实距离；`metric_applied` flag 防叠乘；「改单位 / 微调」分流。
- **贴齐平面**、翻转 Z、绕局部 Z 相对旋转；侧栏欧拉 ↔ 3D 四元数双向同步；保存 `commitPoseForSave()` + `quaternion_wxyz`。
- 导出：`annotator/`、`yolo6d/` **快照改名不覆盖**；导出后自动 `preview_6d.mp4`。

当前最新标注（快照已删，只留最新）：

- `outputs/run1/annotator/`、`outputs/run1/yolo6d/`
- `|Z∠n| ≈ 0°`（以 quat 为准）

## 阶段 D — 全视频 YOLO6D（已完成）

```bash
uv run python scripts/localize_full_video.py --run outputs/run1 --skip_extract
```

- `outputs/run1/yolo6d_full/`：1097/1097 定位成功
- `outputs/run1/yolo6d_full/preview_6d.mp4`

## 阶段 E — 仓库化（已完成，待你首次 commit）

- 目录名：`object_6d_pose_annotation`
- ignore：`data/`、`outputs/`、`.venv/`、`models/*`、`third_party/*`（除脚本）、`__pycache__` 等
- **跟踪** `.cursor/`、`scripts/`、`tools/`、`pyproject.toml`、`uv.lock`
- `scripts/prepare_data.py`：提示/接收 MP4 → 生成 `data/`
- `models/download.sh`：SAM2.1-t（ultralytics assets v8.4.0）
- `third_party/setup.sh`：hloc `@c13273b` + OpenMVS **v2.4.0**
- 已 `git init`（main），**尚未 commit**（按你要求自检后再提交）

## 常用命令

```bash
cd /root/ultralytics/projects/object_6d_pose_annotation
unset VIRTUAL_ENV   # 若仍指向旧 object_sfm/.venv
uv sync
bash third_party/setup.sh
bash models/download.sh

uv run python scripts/prepare_data.py --video /path/to/orbit.mp4
bash scripts/run_sfm.sh
uv run python scripts/prepare_annotator_scene.py --run outputs/run1
uv run python tools/pose_annotator/server.py --run outputs/run1
uv run python scripts/localize_full_video.py --run outputs/run1 --skip_extract
```

## 下一步（可选）

1. 你本地检查后做 **首次 git commit**（必要时加 remote）。
2. 新需求 / 新会话：在 `.cursor/recaps/` **另建**条目，本条不再追加。
3. 若换物体/视频：`prepare_data.py` → SfM → 标注 → `localize_full_video`。

## 续聊方式

精确（完整 log，含本续聊追加段）：

`/root/ultralytics/projects/object_6d_pose_annotation/.cursor/recaps/20260725_object_sfm/transcript.jsonl`

概括：

`/root/ultralytics/projects/object_6d_pose_annotation/.cursor/recaps/20260725_object_sfm/recap.md`

子任务 log：`.../subagents/`
