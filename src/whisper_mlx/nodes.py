import os
import json
from pathlib import Path
from typing import Union, Dict, List, Tuple
import folder_paths
from comfy.utils import ProgressBar
from mlx_whisper import transcribe

# ====================== 字幕常量，沿用你Qwen3‑ASR nodes.py ==================
SUBTITLE_MAX_CHARS = 35
SUBTITLE_MAX_DURATION_S = 6.0
SUBTITLE_SPLIT_PUNCT = "，。！？；"
SUBTITLE_SPLIT_PUNCT_EN = ",.!?;"
TRIM_PUNCT_EXCEPT = "！？!?"

AVAILABLE_SUBTITLE_FORMAT = ["txt", "srt", "vtt", "json", "lrc"]
FLOAT_NONE_VALUE = -999.0
INT_NONE_VALUE = -999

# ------------------------------ 字幕工具函数 ------------------------------
def split_by_text_ratio_interpolate(seg_items, max_chars: int, max_dur: float, puncts: str):
    out = []
    punct_set = set(puncts)
    for seg in seg_items:
        if isinstance(seg, dict):
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            full_text = seg.get("text", "").strip()
        else:
            seg_start = getattr(seg, "start", 0.0)
            seg_end = getattr(seg, "end", 0.0)
            full_text = getattr(seg, "text", "").strip()
        seg_total_dur = seg_end - seg_start
        if not full_text or seg_total_dur <= 1e-6:
            out.append({"start": seg_start, "end": seg_end, "text": full_text})
            continue
        pieces = []
        buf = []
        for ch in full_text:
            buf.append(ch)
            buf_str = "".join(buf)
            if ch in punct_set or len(buf_str) >= max_chars:
                piece = buf_str.strip()
                while len(piece) > 0 and piece[-1] in punct_set and piece[-1] not in TRIM_PUNCT_EXCEPT:
                    piece = piece[:-1].strip()
                if piece:
                    pieces.append(piece)
                buf.clear()
        if buf:
            piece = "".join(buf).strip()
            if piece:
                pieces.append(piece)
        if len(pieces) == 0:
            out.append({"start": seg_start, "end": seg_end, "text": full_text})
            continue
        total_char = sum(len(p) for p in pieces)
        cur_t = seg_start
        for piece in pieces:
            char_cnt = len(piece)
            ratio = char_cnt / total_char
            piece_dur = seg_total_dur * ratio
            p_end = cur_t + piece_dur
            out.append({"start": cur_t, "end": p_end, "text": piece})
            cur_t = p_end
    return out

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{sec:06.3f}".replace(".", ",")

def format_vtt_timestamp(seconds: float) -> str:
    return format_timestamp(seconds).replace(",", ".")

def format_lrc_timestamp(seconds: float) -> str:
    mm = int(seconds // 60)
    ss = seconds % 60
    return f"{mm:02d}:{ss:05.2f}"

def format_transcriptions_to_subtitle(transcriptions: List[Dict], fmt: str,
                                     lrc_meta: Dict = None) -> str:
    lrc_meta = lrc_meta or {}
    if fmt == "txt":
        return "\n".join(s["text"].strip() for s in transcriptions)
    elif fmt == "srt":
        lines = []
        for idx, seg in enumerate(transcriptions, start=1):
            s = format_timestamp(seg["start"])
            e = format_timestamp(seg["end"])
            lines.append(f"{idx}\n{s} --> {e}\n{seg['text'].strip()}\n")
        return "\n".join(lines)
    elif fmt == "vtt":
        lines = ["WEBVTT", ""]
        for idx, seg in enumerate(transcriptions, start=1):
            s = format_vtt_timestamp(seg["start"])
            e = format_vtt_timestamp(seg["end"])
            lines.append(f"{idx}\n{s} --> {e}\n{seg['text'].strip()}\n")
        return "\n".join(lines)
    elif fmt == "json":
        return json.dumps({"segments": transcriptions}, ensure_ascii=False, indent=2)
    elif fmt == "lrc":
        lrc_lines = []
        ti = lrc_meta.get("title","").strip()
        ar = lrc_meta.get("artist","").strip()
        al = lrc_meta.get("album","").strip()
        by = lrc_meta.get("lrc_by","").strip()
        if ti: lrc_lines.append(f"[ti:{ti}]")
        if ar: lrc_lines.append(f"[ar:{ar}]")
        if al: lrc_lines.append(f"[al:{al}]")
        if by: lrc_lines.append(f"[by:{by}]")
        if lrc_lines:
            lrc_lines.append("")
        for seg in transcriptions:
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_text = seg["text"].rstrip("%").strip()
            sub_lines = [ln.strip() for ln in seg_text.splitlines() if ln.strip()]
            if not sub_lines:
                continue
            dur_total = seg_end - seg_start
            line_cnt = len(sub_lines)
            if line_cnt ==1:
                t_str = format_lrc_timestamp(seg_start)
                lrc_lines.append(f"[{t_str}]{sub_lines[0]}")
            else:
                step = dur_total / line_cnt
                for i, sub_text in enumerate(sub_lines):
                    lt = seg_start + step*i
                    t_str = format_lrc_timestamp(lt)
                    lrc_lines.append(f"[{t_str}]{sub_text}")
        return "\n".join(lrc_lines)
    else:
        raise ValueError(f"Unsupported format {fmt}")

def get_incremented_filename(root_out_dir: str, prefix: str, ext: str) -> str:
    """
    prefix支持子目录，如 "audio/tts"
    最终路径: root_out_dir / prefix_{idx}.ext
    """
    prefix_path = Path(prefix)
    sub_dir_part = prefix_path.parent
    file_stem = prefix_path.name

    target_parent = Path(root_out_dir) / sub_dir_part
    target_parent.mkdir(parents=True, exist_ok=True)

    idx = 1
    while True:
        filename = f"{file_stem}_{idx}.{ext}"
        full_path = target_parent / filename
        if not full_path.exists():
            return str(full_path)
        idx += 1

# =========================
# 🔹 MLX‑Whisper 模型加载节点（参考 Qwen3ASRMLXLoader，按本地文件夹路径，带内存缓存）
# =========================
class MLXWhisperLoader:
    _model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_local_dir": ("STRING", {
                    "default": "/path/to/whisper‑large‑v3‑turbo",
                    "tooltip": "本地HF模型文件夹绝对路径，hfd.sh下载的目录，内含config.json"
                }),
            }
        }

    RETURN_TYPES = ("MLX_WHISPER_MODEL",)
    RETURN_NAMES = ("model_handle",)
    FUNCTION = "load_model_node"
    CATEGORY = "MLX‑WHISPER‑ASR"

    def load_model_node(self, model_local_dir):
        key = str(Path(model_local_dir).absolute())
        if key in self._model_cache:
            return (self._model_cache[key],)
        # mlx‑whisper transcribe直接接收本地文件夹路径作为path_or_hf_repo，不需要预加载对象；保存路径句柄
        if not os.path.isdir(key):
            raise NotADirectoryError(f"MLX‑Whisper模型目录不存在：{key}")
        cfg = os.path.join(key, "config.json")
        if not os.path.exists(cfg):
            raise FileNotFoundError(f"模型目录缺少config.json，请检查hfd.sh下载的文件夹: {key}")
        handle = {"local_model_path": key}
        self._model_cache[key] = handle
        print(f"✅ MLX‑Whisper 已缓存本地模型：{key}")
        return (handle,)

# =========================
# 🔹 MLX‑Whisper 转写节点（参考 faster‑whisper节点架构）
# 输入 FILEPATH，输出 MLX_WHISPER_TRANSCRIPTIONS 结构化列表；携带word时间戳
# =========================
class MLXWhisperTranscription:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("FILEPATH", ),
                "model_handle": ("MLX_WHISPER_MODEL", ),
            },
            "optional": {
                "language": ("STRING", {"default": "auto"}),
                "word_timestamps": ("BOOLEAN", {"default": True, "tooltip":"开启返回每个单词start/end时间戳"}),
                "verbose": ("BOOLEAN", {"default": False}),
                "auto_split_subtitle": ("BOOLEAN", {"default": True, "tooltip":"按标点字符插值切分字幕段"}),
            }
        }

    RETURN_TYPES = ("MLX_WHISPER_TRANSCRIPTIONS", )
    RETURN_NAMES = ("transcriptions", )
    FUNCTION = "transcribe"
    CATEGORY = "MLX‑WHISPER‑ASR"

    def transcribe(self,
                   audio: str,
                   model_handle: Dict,
                   language="auto",
                   word_timestamps=True,
                   verbose=False,
                   auto_split_subtitle=True
                   ) -> Tuple[List]:
        local_model_path = model_handle["local_model_path"]
        lang_arg = None if language == "auto" else language

        result = transcribe(
            audio=audio,
            path_or_hf_repo=local_model_path,
            word_timestamps=word_timestamps,
            language=lang_arg,
            verbose=verbose
        )
        raw_segments = result.get("segments", [])
        total_dur = result.get("duration", 0.0)
        pbar = ProgressBar(total_dur)

        if auto_split_subtitle:
            all_punct = SUBTITLE_SPLIT_PUNCT + SUBTITLE_SPLIT_PUNCT_EN
            seg_list = split_by_text_ratio_interpolate(raw_segments,
                                                       max_chars=SUBTITLE_MAX_CHARS,
                                                       max_dur=SUBTITLE_MAX_DURATION_S,
                                                       puncts=all_punct)
        else:
            seg_list = []
            for seg in raw_segments:
                seg_list.append({
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": seg.get("text", "").strip(),
                    "words": seg.get("words", [])
                })
                pbar.update_absolute(seg.get("end",0.0))
        return (seg_list,)

# =========================
# 🔹 MLXWhisperToSubtitle：结构化transcriptions → 字幕文本对象，参考FasterWhisperToSubtitle
# =========================
class MLXWhisperToSubtitle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "transcriptions": ("MLX_WHISPER_TRANSCRIPTIONS",),
                "subtitle_format": (AVAILABLE_SUBTITLE_FORMAT,),
            },
            "optional": {
                "lrc_title": ("STRING", {"default": ""}),
                "lrc_artist": ("STRING", {"default": ""}),
                "lrc_album": ("STRING", {"default": ""}),
                "lrc_by": ("STRING", {"default": "MLX‑Whisper"}),
            }
        }

    RETURN_TYPES = ("MLX_WHISPER_SUBTITLE", )
    RETURN_NAMES = ("subtitle", )
    FUNCTION = "format_to_subtitle"
    CATEGORY = "MLX‑WHISPER‑ASR"

    def format_to_subtitle(self,
                           transcriptions: List[Dict],
                           subtitle_format: str,
                           lrc_title="", lrc_artist="", lrc_album="", lrc_by=""
                           ) -> Tuple[List]:
        lrc_meta = {
            "title": lrc_title,
            "artist": lrc_artist,
            "album": lrc_album,
            "lrc_by": lrc_by
        }
        text = format_transcriptions_to_subtitle(transcriptions, subtitle_format, lrc_meta=lrc_meta)
        subtitle_package = [text, subtitle_format]
        return (subtitle_package,)

# =========================
# 🔹 保存字幕节点：参考SaveSubtitle，输出到comfyui标准output目录，自动递增文件名
# =========================
class MLXWhisperSaveSubtitle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "subtitle": ("MLX_WHISPER_SUBTITLE", )
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "输出根目录；留空使用ComfyUI默认output目录"
                }),
                "prefix": ("STRING", {
                    "default": "mlx_whisper_subtitle",
                    "tooltip": "支持子目录，例如 audio/tts → 在output_dir/audio/下生成 tts_1.json"
                }),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_file_path",)
    FUNCTION = "save_subtitle"
    OUTPUT_NODE = True
    CATEGORY = "MLX-WHISPER-ASR"

    def save_subtitle(self, subtitle: List, output_dir: str, prefix: str):
        sub_text, fmt = subtitle

        # output_dir为空则回退comfy默认output
        if output_dir.strip() == "":
            out_root = folder_paths.get_output_directory()
        else:
            out_root = output_dir.strip()

        out_path = get_incremented_filename(out_root, prefix, fmt)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(sub_text)
        return (out_path,)

# =========================
# 输入文件路径节点，输出 FILEPATH 类型（和faster‑whisper InputFilePath保持一致）
# =========================
class MLXWhisperInputFilePath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filepath": ("STRING", {"default": "", "multiline": False})
            }
        }
    RETURN_TYPES = ("FILEPATH", )
    RETURN_NAMES = ("filepath",)
    FUNCTION = "process_filepath"
    CATEGORY = "MLX‑WHISPER‑ASR"

    def process_filepath(self, filepath):
        fp = Path(filepath).absolute()
        if not fp.exists():
            raise FileNotFoundError(f"音频文件不存在: {fp}")
        return (str(fp),)


NODE_CLASS_MAPPINGS = {
    "MLXWhisperLoader": MLXWhisperLoader,
    "MLXWhisperTranscription": MLXWhisperTranscription,
    "MLXWhisperToSubtitle": MLXWhisperToSubtitle,
    "MLXWhisperSaveSubtitle": MLXWhisperSaveSubtitle,
    "MLXWhisperInputFilePath": MLXWhisperInputFilePath
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXWhisperLoader": "MLX‑Whisper Loader(本地模型目录)",
    "MLXWhisperTranscription": "MLX‑Whisper Transcription",
    "MLXWhisperToSubtitle": "MLX‑Whisper To Subtitle",
    "MLXWhisperSaveSubtitle": "MLX‑Whisper Save Subtitle",
    "MLXWhisperInputFilePath": "MLX‑Whisper Input FilePath"
}
