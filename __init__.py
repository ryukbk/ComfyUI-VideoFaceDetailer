"""ComfyUI-VideoFaceDetailer — selectively upscale/resample small faces in video."""
try:
    # How ComfyUI loads us: as a package, so the sibling module is a relative import.
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # Loaded as a top-level module instead, with no parent package to be relative to.
    # pytest does this - it sees the __init__.py and treats the repo root as a package,
    # but the directory name contains a hyphen so it cannot be imported under its own
    # name - and so do some linters and doc tools. Falling back keeps them working.
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Required for anything in web/ to be served at all: ComfyUI only registers a
# custom node's front-end directory when the package exports WEB_DIRECTORY
# (nodes.py: `if hasattr(module, "WEB_DIRECTORY")`). Without it the scripts are
# simply never loaded, and the widget toggles silently do nothing.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
