# 会话复盘：YOLO6D mask / SAM2 时序 / preview_mask

> 前置：[`20260725_object_sfm`](../20260725_object_sfm/recap.md)（SfM、标注器、yolo6d_full、仓库化）。**本文只记本会话增量。**

## 背景目标

为 `outputs/run1/yolo6d` 与 `yolo6d_full` 在 rgb/labels 之外增加 **mask**，并便于质检；交互 SAM2 的 mask 要按帧落盘，全序列用时序传播生成。

## 关键决策

1. **交互 mask**：`annotator/masks/{stem}.png`（不再只覆盖 `last_mask.png`）；快照携带整个 `masks/`。
2. **全序列 mask**：`SAM2DynamicInteractivePredictor` 时序传播；交互帧作 memory seed；`conf=0.01`（0.25 会在数帧后丢目标）。
3. **去噪**：`clean_binary_mask`（最大连通域、填洞、轻微 close）；交互 dump 与导出均清理。
4. **质检视频**：`preview_mask.mp4`（绿半透明叠加 + 轮廓）；与 `preview_6d.mp4` 一样在导出时自动生成。
5. **环境**：只用项目 `.venv` + `UV_INDEX_URL=清华`；`pyproject.toml` 排除传递依赖 `opencv-python`，保留 headless（无 libGL）。

## 已实现（代码）

| 位置 | 改动 |
|------|------|
| `tools/pose_annotator/segment.py` | `clean_binary_mask`、`segment_sam2_bbox`、`propagate_masks_temporal`、保存时去噪 |
| `tools/pose_annotator/server.py` | `/api/segment` → dump `masks/{stem}.png`；`GET /api/mask` 加载 |
| `tools/pose_annotator/static/app.js` | 切帧加载已存 mask；状态提示 preview_mask |
| `tools/pose_annotator/export_bridge.py` | 导出写 `mask/`；`ensure_yolo6d_mask_preview`；rgb 空时用 scene 回退 |
| `scripts/localize_full_video.py` / `export_yolo6d_full_video.py` | 写 mask + 自动 `preview_6d` / `preview_mask` |
| `scripts/export_masks_sam2.py` | **新建**：清洗交互 dump → 时序传播 → 写 mask + preview |
| `pyproject.toml` / `uv.lock` | `exclude-dependencies = ["opencv-python"]` |

## 产物状态（run1）

- 交互 seed：`annotator/masks/` → `frame_000000/037/070/072.png`（已清洗，单连通域）
- `yolo6d/mask/`：74；`yolo6d_full/mask/`：1097
- 预览：`yolo6d/preview_mask.mp4`、`yolo6d_full/preview_mask.mp4`
- 注意：`yolo6d/rgb/` 当前可能为空（旧 symlink）；preview_mask 可回退 `annotator/scene.json` 图像路径

## 常用命令

```bash
cd /root/ultralytics/projects/object_6d_pose_annotation
unset VIRTUAL_ENV
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 标注器（重新点 FG 会按帧 dump masks/）
uv run python tools/pose_annotator/server.py --run outputs/run1

# 仅回填 / 刷新 mask + preview_mask（时序 SAM2）
uv run python scripts/export_masks_sam2.py --run outputs/run1
# 试跑：--max_frames 40  --targets yolo6d
```

## 已知问题 / 可选后续

1. 时序偶发空 mask：已加「上一帧 mask 重种子」recovery；全量 recovery 重跑曾被中断，若个别帧漂移可再跑 `export_masks_sam2.py`。
2. 标注器保存导出仍会走逐帧 `resolve_frame_mask`（优先交互 dump，否则 SAM2 box）；全序列质检主路径以 `export_masks_sam2.py` / localize 导出为准。
3. 本会话改动 **尚未 git commit**（按既有约定等你确认）。

## 续聊

- 概括：本目录 `recap.md`
- 精确：本目录 `transcript.jsonl`（会话 id `4784ba15-…`）
- 前置：`../20260725_object_sfm/`
