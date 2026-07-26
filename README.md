# object_6d_pose_annotation

手机绕拍静物视频 → SfM（hloc: SuperPoint + LightGlue）→ 已知长度定米制尺度 → 浏览器标注物体 6D 框 → 导出 YOLO6D 标签（含全视频定位）。

## 目录

```
object_6d_pose_annotation/
  data/                         # gitignore — 用 scripts/prepare_data.py 生成
  models/download.sh            # 拉取 SAM2 权重（权重本身 gitignore）
  outputs/                      # gitignore — SfM / 标注 / YOLO6D 产物
  scripts/                      # 抽帧、SfM、导出、全视频定位等
  tools/pose_annotator/         # 浏览器 6D 标注器
  third_party/setup.sh          # 拉取 hloc 源码树 + OpenMVS 预编译包
  .cursor/recaps/               # 会话复盘（入库，便于续聊与优化）
```

## 环境

系统依赖：`ffmpeg`、`colmap`（稠密 CUDA MVS 另需 CUDA 版 COLMAP，可选）、`curl`、`git`、`unzip`、[`uv`](https://github.com/astral-sh/uv)。

```bash
cd /path/to/object_6d_pose_annotation
uv sync                          # 按 pyproject.toml / uv.lock 建 .venv（含 torch cu124、hloc、lightglue）
bash third_party/setup.sh        # Hierarchical-Localization @ c13273b + OpenMVS v2.4.0
bash models/download.sh          # models/sam2.1_t.pt
```

之后用 `uv run …` 或 `source .venv/bin/activate`。

### third_party 钉扎版本

| 组件 | 版本 |
|------|------|
| [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization) | commit `c13273bd0ecc2917a35910fd843712a1c6243193` |
| [LightGlue](https://github.com/cvg/LightGlue)（uv 依赖） | commit `eb42fee2d71449efb0aa5c10549752b5d75384d8` |
| [OpenMVS](https://github.com/cdcseacave/openMVS/releases/tag/v2.4.0) | `v2.4.0` / `OpenMVS_Ubuntu_x64.zip` |
| SAM2.1-t | [ultralytics assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt) |

## 准备 data/

`data/` 整目录不入库。自备绕拍 **MP4**，生成与当前流水线一致的目录：

```bash
# 交互：提示输入 mp4 路径
uv run python scripts/prepare_data.py

# 或显式指定
uv run python scripts/prepare_data.py --video /path/to/your_orbit.mp4
```

会得到：

- `data/<name>.mp4` — 复制进来的原视频  
- `data/frames/` — 约 2 fps、最长边 1600（默认 SfM / run1）  
- `data/frames_native/` — 同采样、原生分辨率（HQ / run2）  
- `data/extract_meta.json`

## 推荐流程

### 1) SfM（默认 run1：1600 / SuperPoint+LightGlue）

```bash
bash scripts/run_sfm.sh
# 或: uv run python scripts/run_sfm_hloc.py --images data/frames --outputs outputs/run1
```

### 2) 导出（相对尺度）

```bash
uv run python scripts/export_poses.py \
  --sfm_dir outputs/run1/sfm \
  --out_dir outputs/run1/export
```

### 3) 已知长度定尺度

在点云上取真实长度对应的两点（例如物体高度 0.185 m）：

```bash
uv run python scripts/apply_scale.py \
  --sfm_dir outputs/run1/sfm \
  --out_dir outputs/run1/metric \
  --p1 x1 y1 z1 --p2 x2 y2 z2 \
  --real_length_m 0.185
```

也可在标注器内做尺度标定。

### 4)（可选）稠密重建

```bash
bash scripts/run_dense_colmap_cuda.sh   # CUDA COLMAP MVS
# 或: bash scripts/run_dense.sh        # 深度融合备选
```

### 5) 浏览器标注 → YOLO6D

```bash
uv run python scripts/prepare_annotator_scene.py --run outputs/run1
uv run python tools/pose_annotator/server.py --run outputs/run1
# 浏览器打开提示的 URL，摆好物体框后保存
```

### 6) 全视频 6D 标签

```bash
uv run python scripts/localize_full_video.py --run outputs/run1 --skip_extract
# 输出: outputs/run1/yolo6d_full/ （含 preview_6d.mp4）
```

## 会话复盘（续聊）

仓库内保留 Cursor 复盘，便于管理与续作：

- 概括：`.cursor/recaps/20260725_object_sfm/recap.md`
- 完整 log：`.cursor/recaps/20260725_object_sfm/transcript.jsonl`
- 元数据：`.cursor/recaps/20260725_object_sfm/meta.json`

新对话中贴上述路径并说明「续聊 / 恢复」即可。

## 与 OnePose++ 的关系

| | 本仓库 | OnePose++ |
|--|--|--|
| 位姿来源 | SfM（无需 ARKit） | ARKit + BA / 或已有位姿 |
| 匹配 | SuperPoint+LightGlue（hloc） | LoFTR 关键点无关 SfM |
| 用途 | 给**本段视频**打 6D + 点云 | 建库后估**新视频** pose |

## 输出含义

- `K.txt`：内参  
- `poses_w2c/*.txt`：world-to-camera  
- `sparse.ply` / 稠密点云：MeshLab / CloudCompare 可查看  
- YOLO6D：`yolo6d/`（关键帧）与 `yolo6d_full/`（全视频）
