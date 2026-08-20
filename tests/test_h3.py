import os
import sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nodes import FaceTrackCropAndGate, FaceTrackPasteBack, _next_valid_clip_len

ok = True
def check(n, c):
    global ok; print(("PASS " if c else "FAIL ") + n); ok = ok and c

# ── length grid helper ──────────────────────────────────────────────────────
check("ltx grid 8n+1", [_next_valid_clip_len(n, "ltx") for n in (1, 2, 9, 10)] == [1, 9, 9, 17])
check("h3 grid 17k+5", [_next_valid_clip_len(n, "minimax_h3") for n in (1, 5, 6, 22, 23)] == [5, 5, 22, 22, 39])

# ── build a clip whose tracked face is small (< 10% of width) on 3 frames ────
W, H, N = 400, 200, 3
imgs = torch.rand(N, H, W, 3)
mt = torch.zeros(N, H, W)
for i in range(N):
    # 20px face (5% of width) centred; well under the 10% width gate
    cy, cx = 100, 200
    mt[i, cy - 10:cy + 10, cx - 10:cx + 10] = 1.0

crop = FaceTrackCropAndGate()
face_clip, data, target_size, n_real, n_runs = crop.crop(
    imgs, mt, upscale_ratio=2.0, threshold_type="width",
    max_width_fraction=0.10, max_height_fraction=0.10, max_area_percent=10.0,
    hysteresis=0.02, padding=0.3, smooth_alpha=1.0, max_size_deviation=0.5,
    size_smooth_alpha=0.4, min_threshold_percent=0.0, resampler="minimax_h3")

clip_len = face_clip.shape[0]
check("h3: clip padded onto 17k+5 grid", clip_len == _next_valid_clip_len(n_real, "minimax_h3") and (clip_len - 5) % 17 == 0)
check("h3: data records resampler", data.get("resampler") == "minimax_h3")
check("h3: data clip_length matches", data.get("clip_length") == clip_len and data.get("ltx_length") == clip_len)
present = [e for e in data["entries"] if e.get("present")]
check("h3: present entries carry face_px", len(present) == n_real and all(e.get("face_px", 0) > 0 for e in present))
check("h3: entries count == clip length (paste 1:1)", len(data["entries"]) == clip_len)

# default resampler stays LTX (backward compatible)
_, data_ltx, _, nr2, _ = crop.crop(
    imgs, mt, 2.0, "width", 0.10, 0.10, 10.0, 0.02, 0.3, 1.0)
check("default resampler is ltx", data_ltx.get("resampler") == "ltx" and (data_ltx["clip_length"] - 1) % 8 == 0)

# ── paste back with colour_match on (must run and match count/shape) ─────────
paste = FaceTrackPasteBack()
processed = torch.rand(clip_len, target_size, target_size, 3)  # pretend H3 output
(out,) = paste.paste(imgs, processed, data, feather=0.15, blend_mode="mask",
                     only_present_frames=True, colour_match=1.0)
check("paste colour_match keeps frame shape", tuple(out.shape) == (N, H, W, 3))
check("paste colour_match stays in [0,1]", float(out.min()) >= 0.0 and float(out.max()) <= 1.0)

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
