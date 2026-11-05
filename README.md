# comfyui‑whisper‑mlx
ComfyUI 自定义节点，基于 **mlx‑whisper**，在 Apple Silicon(Mac mini M4/M3/M2) 本地完成语音识别ASR，支持单词级时间戳，读取本地HF模型权重。

> ⚠️ **仅支持 Apple Silicon macOS**，依赖 Metal / MLX 框架，Windows/Linux 无法运行。

## ✨ 功能特性
1. **MLX‑Whisper Loader(本地模型目录)**
    - 直接指定本地HF模型文件夹路径（使用 `hfd.sh` 下载的权重），内存缓存模型句柄，避免重复加载。
2. **MLX‑Whisper Transcription**
    - 音频转写，支持开关 `word_timestamps`，输出**单词粒度时间戳**（仅json输出保留单词时间）。
    - 自动字幕切分：按中文/英文标点+字符数量插值切分字幕片段。
    - 支持语言 auto / zh / en / ja / ko。
3. **MLX‑Whisper To Subtitle**
    - 将结构化转写结果输出为字幕文本：`txt / srt / vtt / json / lrc`。
    - LRC 格式支持标题、艺术家、专辑、生成工具元标签。
4. **MLX‑Whisper Save Subtitle**
    - 可自定义输出根目录；留空自动使用 ComfyUI 默认 output 目录。
    - prefix 支持子目录层级，例如 `batch01/audio_001`，自动创建子文件夹。
    - 文件名自动数字递增，防止覆盖旧文件。
5. **MLX‑Whisper Input FilePath**
    - 输入音频绝对路径，输出 FILEPATH 类型供转写节点使用。

> json 格式是唯一保留**单词级时间戳**的输出格式；srt/vtt/lrc只保存segment片段字幕。

## 📦 环境依赖
需要在 ComfyUI 的虚拟环境中安装依赖：
```bash
# 激活ComfyUI虚拟环境
source .venv/bin/activate

pip install mlx-whisper==0.4.3
```

## 📥 模型权重下载（本地，不走HuggingFace在线下载）
使用 hfd.sh 下载 `mlx‑community/whisper‑large‑v3‑turbo` 到本地文件夹：
```bash
hfd.sh mlx-community/whisper-large-v3-turbo --local-dir ./whisper-large-v3-turbo
```
> 文件夹内必须包含 `config.json`、`model.safetensors`，将此文件夹**绝对路径**填入 `MLX‑Whisper Loader` 的 `model_local_dir`。

## 🧩 ComfyUI 工作流连线示例
```
MLX‑Whisper Input FilePath 填入音频绝对路径
            ↓ FILEPATH
MLX‑Whisper Transcription ← MLX‑Whisper Loader(model_handle)
            ↓ transcriptions
MLX‑Whisper To Subtitle（选择字幕格式，配置LRC元信息）
            ↓ subtitle
MLX‑Whisper Save Subtitle（设置 output_dir / prefix，prefix支持子目录如 `asr/batch01`）
```

## 📝 节点参数说明
### MLX‑Whisper Transcription
- `audio_path`：音频文件绝对路径（wav/mp3/m4a）
- `word_timestamps`：开启后结果中携带每个单词的 `start/end` 时间戳，json输出可见
- `auto_split_subtitle`：开启按标点+字符占比插值切分长字幕行
- `language`：`auto`自动检测，也可指定 `zh` / `en` / `ja` / `ko`

### MLX‑Whisper Save Subtitle
- `output_dir`：输出根目录，留空使用ComfyUI默认`output`目录；支持绝对路径
- `prefix`：支持子目录，示例：`output_asr/group01` → 在输出根目录下生成 `output_asr/group01_1.srt`，自动创建不存在的子目录

## 📂 目录结构
```
comfyui_whisper_mlx/
 ├─ __init__.py
 ├─ src
 |   └─ whisper_mlx
 |      └─ nodes.py
 └─ pyproject.toml
```

## 🛠️ 手动安装
1. 将整个项目文件夹放到 `ComfyUI/custom_nodes/`
2. 激活 ComfyUI venv，安装依赖 `pip install mlx‑whisper==0.4.3`
3. 重启 ComfyUI，节点分类：`MLX‑WHISPER‑ASR`

> ComfyUI‑Manager 可识别本扩展，通过管理器安装同样需要自行下载模型权重到本地，本节点**不会自动下载模型**。

## ⚠️ 注意事项
1. **只在 Apple Silicon Mac 工作**；Intel Mac、Windows、Linux 不支持 mlx‑whisper。
2. large‑v3‑turbo 建议至少 16GB 统一内存；内存紧张可换 smaller 系列mlx‑whisper权重。
3. 模型权重需要手动hfd.sh下载本地，节点不会在线拉取权重。
4. 单词时间戳信息只保存在json输出，srt/vtt/lrc只有segment级时间。

---

## License
MIT
