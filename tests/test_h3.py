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
face_clip, data, target_size, n_real, n_runs, frame_count, enhanced, report = crop.crop(
    imgs, mt, upscale_ratio=2.0, threshold_type="width", max_threshold_percent=10.0,
    hysteresis_percent=2.0, padding=0.3, smooth_alpha=1.0, max_size_deviation=0.5,
    size_smooth_alpha=0.4, min_threshold_percent=0.0, resampler="minimax_h3")

clip_len = face_clip.shape[0]
check("h3: clip padded onto 17k+5 grid", clip_len == _next_valid_clip_len(n_real, "minimax_h3") and (clip_len - 5) % 17 == 0)
check("h3: data records resampler", data.get("resampler") == "minimax_h3")
check("h3: data clip_length matches", data.get("clip_length") == clip_len and data.get("ltx_length") == clip_len)
present = [e for e in data["entries"] if e.get("present")]
check("h3: present entries carry face_px", len(present) == n_real and all(e.get("face_px", 0) > 0 for e in present))
# H3FaceRefine reads these to ramp per-frame denoise between the gate's thresholds.
check("h3: track_data carries gate window", data.get("threshold_type") == "width"
      and abs(data.get("max_threshold_frac", -1) - 0.10) < 1e-9
      and data.get("min_threshold_frac") == 0.0)
check("h3: present entries carry measure_frac (threshold_type units)",
      all(e.get("measure_frac", 0) > 0 for e in present))
check("h3: entries count == clip length (paste 1:1)", len(data["entries"]) == clip_len)

# default resampler stays LTX (backward compatible)
_, data_ltx, _, nr2, _, fc_ltx, enh_ltx, _ = crop.crop(
    imgs, mt, 2.0, "width", 10.0, 2.0, 0.3, 1.0)
check("default resampler is ltx", data_ltx.get("resampler") == "ltx" and (data_ltx["clip_length"] - 1) % 8 == 0)
# frame_count output == the padded clip length (drives the resampler `length` directly)
check("frame_count == clip length", frame_count == face_clip.shape[0] == data["clip_length"]
      and fc_ltx == data_ltx["clip_length"])
# enhanced BOOLEAN drives LazySwitchKJ: True when >=1 frame qualified.
check("enhanced=True when frames qualify", enhanced is True and n_real > 0)
# report STRING gives width/height/area fraction ranges to help pick thresholds.
check("report lists width/height/area ranges",
      isinstance(report, str) and all(k in report for k in ("width", "height", "area"))
      and "min–max" in report and "enhanced" in report)

# ── 0 frames qualify -> graceful no-op dummy clip (must NOT raise) ───────────
# (a) large face only: every face is ABOVE the width threshold.
big = torch.zeros(N, H, W)
for i in range(N):
    big[i, 40:160, 100:300] = 1.0  # 200px face = 50% of width, well above 10%
noop_clip, noop_data, noop_size, noop_real, noop_runs, noop_fc, noop_enh, noop_rep = crop.crop(
    imgs, big, upscale_ratio=2.0, threshold_type="width", max_threshold_percent=10.0,
    hysteresis_percent=0.0, padding=0.3, smooth_alpha=1.0, max_size_deviation=0.5,
    size_smooth_alpha=0.4, min_threshold_percent=0.0, resampler="minimax_h3")
check("no-op: 0 real frames, does not raise", noop_real == 0 and noop_runs == 0)
# enhanced=False -> LazySwitchKJ skips the branch, so the dummy never hits Resize.
check("no-op: enhanced=False (switch skips branch)", noop_enh is False)
# target_size (slot 2, what Resize width/height MUST read) is floored >= 8 so a
# no-face clip never feeds 0 into ImageResizeKJv2 ("height and width must be > 0").
check("no-op: target_size floored >= 8 (never 0)", noop_size >= 8)
check("no-op: frame_count == dummy clip length (>0)", noop_fc == noop_clip.shape[0] and noop_fc >= 5)
check("no-op: dummy clip on h3 grid", (noop_clip.shape[0] - 5) % 17 == 0 and noop_clip.shape[0] >= 5)
check("no-op: all entries present=False", all(not e.get("present") for e in noop_data["entries"]))
# (b) empty enable window: min_threshold == threshold (the reported bug config).
_, ew_data, _, ew_real, _, _, ew_enh, _ = crop.crop(
    imgs, mt, upscale_ratio=2.0, threshold_type="width", max_threshold_percent=10.0,
    hysteresis_percent=0.0, padding=0.3, smooth_alpha=1.0, max_size_deviation=0.5,
    size_smooth_alpha=0.4, min_threshold_percent=10.0, resampler="minimax_h3")
check("no-op: empty enable window (min==max) yields 0 real, no raise", ew_real == 0 and ew_enh is False)
# paste-back on a no-op clip is a true passthrough (original unchanged).
paste_noop = FaceTrackPasteBack()
dummy_proc = torch.rand(noop_clip.shape[0], noop_size, noop_size, 3)
(pass_out,) = paste_noop.paste(imgs, dummy_proc, noop_data, feather=0.15,
                               blend_mode="mask", only_present_frames=True,
                               colour_match=0.0)
check("no-op: paste-back returns original unchanged", torch.equal(pass_out, imgs))

# ── paste back with colour_match on (must run and match count/shape) ─────────
paste = FaceTrackPasteBack()
processed = torch.rand(clip_len, target_size, target_size, 3)  # pretend H3 output
(out,) = paste.paste(imgs, processed, data, feather=0.15, blend_mode="mask",
                     only_present_frames=True, colour_match=1.0)
check("paste colour_match keeps frame shape", tuple(out.shape) == (N, H, W, 3))
check("paste colour_match stays in [0,1]", float(out.min()) >= 0.0 and float(out.max()) <= 1.0)

# ── FaceTrackAudioSlice: audio reindexed to the gated clip frames ────────────
from nodes import FaceTrackAudioSlice
sr = 16000
aud = {"waveform": torch.zeros(1, 2, sr * 3), "sample_rate": sr}   # 3s stereo
out_aud, _rep = FaceTrackAudioSlice().slice(aud, data, source_fps=24.0, target_fps=24.0)
win = round(sr / 24.0)
check("audio slice length == clip_frames * (sr/target_fps)",
      out_aud["waveform"].shape[-1] == len(data["entries"]) * win)
check("audio slice keeps sample_rate + channels",
      out_aud["sample_rate"] == sr and out_aud["waveform"].shape[:2] == (1, 2))
# empty/missing audio -> passthrough, never crashes
pass_aud, _ = FaceTrackAudioSlice().slice({"waveform": None, "sample_rate": 0}, data)
check("audio slice tolerates missing waveform", pass_aud is not None)

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
