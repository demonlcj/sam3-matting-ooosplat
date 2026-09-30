# SAM3 OOOSplat Video Tracker

这是一个用于 OOOSplat 前处理的本地视频主体跟踪工具。当前版本只负责：

1. 选择视频；
2. 在首帧点击主体内部一点（左键前景，右键背景）；
3. 使用 `F:\SAM3` 中的 SAM3 checkpoint 跟踪整段视频；
4. 导出每帧二值 Mask、Mask 叠加预览视频，以及带透明背景的 PNG 序列。

Matting Anything 已作为项目内子目录接入可选的逐帧 Alpha 细化阶段。程序默认查找 `Matting-Anything\checkpoints\mam_sam_vitb.pth`，也可以在右侧 Matting 面板中选择其他 `.pth` 权重。

## 工作台界面

- 顶部显示视频来源、帧数、FPS 和当前处理状态；
- 左侧显示 Video、SAM3、Matting、Export 四阶段流程；
- 中央预览可切换原图、Mask、Alpha 和叠加显示；
- 右侧显示当前帧和提示点信息，并提供 Matting Anything MAM ViT-B、权重路径和边缘宽度控制；
- 底部提供可点击帧缩略图、播放控制、帧定位和任务进度；
- SAM3 完成后只保存 Mask 文件路径，预览时按需载入，避免长视频占用大量内存。

Matting 阶段读取 `mask_frames`，以每帧 Mask 的目标框作为 MAM 提示，并在 SAM3 Mask 的边缘区域细化 Alpha。输出位于结果包的 `matting_output` 子目录。

Matting 完成后，底部预览源下拉框可以在 `SAM3 Mask` 和 `Matting Alpha` 之间切换；两套输出都会保留，不会覆盖 SAM3 的 `rgba_video.webm`。

右侧 `OOOSPLAT` 标签页可以把当前成果转换为 OOOSplat 0.5.0 可直接读取的透明 PNG 图片序列。默认以 8 FPS 采样并优先使用 Matting Alpha；没有 Matting 结果时自动回退到 SAM3 Mask。可调整采样 FPS、最大边长和透明截止值，每次导出都会创建新的 `ooosplat_input` 文件夹，不覆盖已有导出。

## 启动

默认使用 `F:\SAM3\.venv_desktop\Scripts\python.exe` 和 `F:\SAM3\sam3.pt`：

```powershell
cd F:\SAM3_OOOSplat
START_SAM3_OOOSPLAT.bat
```

也可以显式指定路径：

```powershell
$env:SAM3_ROOT = 'F:\SAM3'
$env:SAM3_PYTHON = 'F:\SAM3\.venv_desktop\Scripts\python.exe'
$env:SAM3_CHECKPOINT = 'F:\SAM3\sam3.pt'
python app.py
```

`Matting-Anything` 默认按 `app.py` 所在目录自动定位，不再依赖固定盘符。其模型权重约 389 MB，已通过 `.gitignore` 排除；克隆仓库后需要把 `mam_sam_vitb.pth` 放入 `Matting-Anything\checkpoints`，或者通过界面重新选择权重。

透明视频编码需要系统可找到 `ffmpeg.exe`；如果没有安装 FFmpeg，程序仍会完整输出 `rgba_frames` 透明 PNG 序列，不会影响后续 Matting Anything。

界面中的“输出位置”可以选择结果保存的父目录。程序会在该目录下创建一个独立任务文件夹，默认名称为 `<视频名>_sam3_output`；如果同名文件夹已存在，会自动创建 `_2`、`_3` 等新文件夹，不会覆盖之前的结果：

```text
mask_frames\\000000.png       # COLMAP 可用的白色前景/黑色背景 Mask
rgba_frames\\000000.png       # 透明 PNG，后续可接 Matting Anything
rgba_video.webm                # VP9 + Alpha 透明视频（优先）
rgba_video.mov                 # FFmpeg 回退的 QTRLE + Alpha 视频
overlay.mp4                   # 原视频 + Mask 预览（无 Alpha）
metadata.json
sam3_ooosplat_error.log      # 若 pythonw 发生异常，记录本次任务的错误

matting_output/matting_frames/000000.png       # Matting Alpha
matting_output/matting_rgba_frames/000000.png # Matting 后的 RGBA
matting_output/rgba_video.webm                 # Matting 后透明视频
matting_output/matting_metadata.json

ooosplat_input/images/000000.png   # 在 OOOSplat 中选择此图片序列目录
ooosplat_input/alpha/000000.png    # 保留软 Alpha
ooosplat_input/masks/000000.png    # 二值前景 Mask
ooosplat_input/manifest.json       # 采样参数与源帧映射
```

## 使用说明

- 打开视频后，首帧会显示在左侧；
- 点击顶部“打开成果”可以重新载入已有的 `*_sam3_output` 文件夹；直接选择其 `matting_output` 子目录也可以自动识别根目录；
- 左键点击手办内部添加前景点；右键点击背景添加背景点；
- 点击“运行 SAM3 跟踪”；
- 结果会保存到输出目录，并在右侧显示进度和结果预览；
- 已有成果会恢复 SAM3 Mask、Matting Alpha、时间轴、提示点和播放预览；原视频缺失时自动使用 `rgba_frames` 回退预览；
- 在右侧切换到 `OOOSPLAT`，设置 Alpha 来源与采样参数后点击“生成 OOOSplat 输入”；完成后在 OOOSplat 中选择弹窗所示的 `ooosplat_input\images` 文件夹；
- 如果首帧选错，重新点击会清空旧提示并使用新提示。

长视频或高分辨率视频可能需要较多显存。建议先用 10--30 秒素材验证流程。
