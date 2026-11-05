"""Top-level package for whisper_mlx."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """comfyui_whisper_mlx"""
__email__ = "feishengfei@126.com"
__version__ = "0.0.1"

from .src.whisper_mlx.nodes import NODE_CLASS_MAPPINGS
from .src.whisper_mlx.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
