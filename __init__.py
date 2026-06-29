"""ComfyUI-VideoFaceDetailer — selectively upscale/resample small faces in video."""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Frontend extension: shows only the active threshold widget (width/height/area).
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
