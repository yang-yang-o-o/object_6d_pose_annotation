# 会话复盘：FoundationPose → object_6d_pose_annotation

## 背景目标

- 原工作区：`/root/FoundationPose`（uv 环境跑通官方 demo，无头保存可视化）。
- 新目标：用手机绕拍静物，**无 ARKit**，经 SfM + 已知长度定尺度，给帧打 6D，供后续 YOLO6D 训练。
- 当前工程：`/root/object_6d_pose_annotation`（续聊请切到此目录）。

## 关键决策

1. **标注单目/本段绕拍视频**：不必硬走 BundleSDF→FoundationPose；FoundationPose 推理需要 depth。
2. **OnePose++** 更贴「扫物体 + 标 RGB 视频」，但官方依赖 iOS Cap/ARKit；无 iOS 时用 **SfM + 已知尺寸定尺度** 替代位姿与米制。
3. 只标本段建库视频时，**COLMAP/hloc SfM 位姿本身就是标签**；OnePose++ 更适合「建一次模型、批量标新视频」。
4. 重建质量：采用 **hloc = SuperPoint + LightGlue + COLMAP**（对齐 OnePose 系深度特征思路），而非默认 SIFT。

## FoundationPose 侧（已完成，可参考 docs/）

- 路径：`/root/FoundationPose`，venv：`.venv`，GPU：3080 Ti 12GB。
- `run_demo.py`：无 `imshow`，保存 `track_vis/*.png` + mp4；支持 `--shorter_side`（driller 需 480 防 OOM）。
- 文档：`docs/reproduction.md`、`docs/changelog_local.md`。

## object_6d_pose_annotation 当前状态

| 项 | 内容 |
|----|------|
| 视频 | `data/VID_20260725_165829.mp4`（自 `~/VID_...` 移入），3840×2160，~1097 帧 |
| 抽帧 | `data/frames/`，74 张，1600×900，~2 fps |
| SfM | `outputs/run1/`，**74/74 注册**，8447 点，reproj≈1.33 px |
| 导出 | `outputs/run1/export/`：`K.txt`、`obj_in_cam/`、`sparse.ply`、`camera.json` |
| 稠密 | apt COLMAP 无 CUDA，`patch_match_stereo` 失败；打 6D 不依赖稠密 |
| 尺度 | **未做**；需 MeshLab 点两点 + `apply_scale.py --real_length_m` |

### 常用命令

```bash
cd ~/object_6d_pose_annotation && source .venv/bin/activate

# 已跑过的 SfM
bash scripts/run_sfm.sh

# 定尺度（填真实长度与两点）
python scripts/apply_scale.py \
  --sfm_dir outputs/run1/sfm \
  --out_dir outputs/run1/metric \
  --p1 x1 y1 z1 --p2 x2 y2 z2 \
  --real_length_m H
```

## 下一步（恢复后优先）

1. 用户量物体真实高度/边长，在 `sparse.ply` 上取对应两点 → 跑 `apply_scale.py`。
2. 定义物体坐标系（原点/轴向），把 `obj_in_cam` 变到约定物体框。
3. （可选）提高抽帧：`--max_side 2400 --fps 3` 重跑 SfM。
4. （可选）CUDA COLMAP / OpenMVS 做稠密。
5. 导出 YOLO6D 标签格式；需要多背景时再拍 test 序列。

## 续聊方式

在 Cursor 打开 **`/root/object_6d_pose_annotation`**，并说明：

> 精确续聊：`/root/object_6d_pose_annotation/.cursor/recaps/20260725_object_sfm/transcript.jsonl`  
> 或概括续聊：`.../recap.md`

快捷链接：`~/object_6d_pose_annotation_recap_latest` → 本目录。
