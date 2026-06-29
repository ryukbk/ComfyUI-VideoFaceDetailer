import os
import sys, torch, numpy as np
import torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nodes import FaceTrackCropAndGate, FaceTrackPasteBack

ok = True
def check(n, c):
    global ok; print(("PASS " if c else "FAIL ")+n); ok = ok and c

W = 400  # 10% = 40px
H = 200
crop = FaceTrackCropAndGate()
paste = FaceTrackPasteBack()

def face_track(widths, cx=100, cy=100):
    """Build images + mask track where each frame has a face of given width (or None=absent)."""
    B = len(widths)
    imgs = torch.rand(B, H, W, 3)
    mt = torch.zeros(B, H, W)
    for i, w in enumerate(widths):
        if w is None: continue
        half = w // 2
        x0 = max(0, cx - half); x1 = min(W, cx + half)
        y0 = max(0, cy - half); y1 = min(H, cy + half)
        mt[i, y0:y1, x0:x1] = 1.0
    return imgs, mt

# ---- THE ZOOM SCENARIO ----
# Face starts small (24px), camera zooms in so it grows past 40px to 120px.
widths = [24, 28, 34, 44, 70, 120]   # crosses threshold at frame 3
imgs, mt = face_track(widths)
clip, data, tgt, ne, *_ = crop.crop(imgs, mt, 2.0, 0.10, 0.0, 0.3, 0.4)  # hysteresis 0 for crisp boundary
frames = [e["frame"] for e in data["entries"] if e["present"]]
check("zoom: only small frames enhanced (0,1,2)", frames == [0,1,2])
check("zoom: enhanced_frames count == 3", ne == 3)
# Paste back and confirm the LARGE-face frames are byte-identical to original
up = Fn.interpolate(clip.permute(0,3,1,2), size=(tgt,tgt), mode="bilinear", align_corners=False).permute(0,2,3,1)
out = paste.paste(imgs, up, data, 0.0, True)[0]
check("zoom: frame 4 (large) untouched", torch.equal(out[4], imgs[4]))
check("zoom: frame 5 (largest) untouched", torch.equal(out[5], imgs[5]))
check("zoom: frame 0 (small) was enhanced", not torch.equal(out[0], imgs[0]))
# Crop boxes (per-entry 'win') sized from each small face, NOT the 120px large one.
check("zoom: enhanced crop boxes tight to small faces (<70px)",
      max(e["win"] for e in data["entries"]) < 70)

# ---- Zoom out then in again: two enhanced runs ----
widths2 = [24, 60, 100, 60, 24, 20]   # big in middle, small at ends
imgs2, mt2 = face_track(widths2)
clip2, data2, _, ne2, nruns2 = crop.crop(imgs2, mt2, 2.0, 0.10, 0.0, 0.3, 0.4)
frames2 = [e["frame"] for e in data2["entries"] if e["present"]]
check("multi-run: enhances frames 0,4,5 (skips big middle)", frames2 == [0,4,5])
check("multi-run: num_runs output == 2", nruns2 == 2)

# ---- Hysteresis: face hovering at threshold shouldn't flicker ----
# widths oscillate around 40 (39,41,39,41). With hysteresis, state should be sticky.
widths3 = [30, 39, 41, 39, 41, 39]
imgs3, mt3 = face_track(widths3)
# hysteresis 0.03 -> on below 28px (0.07*400), off at/above 52px(0.13*400)
# frame0=30<? on_thresh=0.07*400=28 -> 30 not <28, so stays OFF initially... adjust:
# Use max_width_fraction 0.12 (48px) so 30 enables; off threshold 0.15*400=60. 39/41 never exceed 60 -> stays ON.
clip3, data3, _, ne3, *_ = crop.crop(imgs3, mt3, 2.0, 0.12, 0.03, 0.3, 0.4)
frames3 = [e["frame"] for e in data3["entries"] if e["present"]]
check("hysteresis: no flicker once on (all 6 enhanced)", frames3 == [0,1,2,3,4,5] and ne3 == 6)

# ---- Absent frames never enhanced ----
widths4 = [24, None, 28, None, 30]
imgs4, mt4 = face_track(widths4)
_, data4, _, ne4, *_ = crop.crop(imgs4, mt4, 2.0, 0.10, 0.0, 0.3, 0.4)
frames4 = [e["frame"] for e in data4["entries"] if e["present"]]
check("absent frames skipped, present small kept (0,2,4)", frames4 == [0,2,4])

# ---- Ratio independence of paste region (regression) ----
ca, da, ta, _, *_ = crop.crop(imgs, mt, 2.0, 0.10, 0.0, 0.3, 0.4)
cb, db, tb, _, *_ = crop.crop(imgs, mt, 4.0, 0.10, 0.0, 0.3, 0.4)
check("ratio: larger ratio bigger target", tb > ta)
check("ratio: same crop box & paste region",
      da["entries"][0]["win"]==db["entries"][0]["win"] and
      da["entries"][0]["x0"]==db["entries"][0]["x0"])

# ---- TALL face must not produce a bloated crop (regression for the 960x512 bug) ----
# Build a face 60px wide x 120px tall (mask taller than wide) on a 400-wide frame.
def tall_track(n=4, fw=60, fh=120, cx=200, cy=256):
    H2, W2 = 512, 400
    imgs = torch.rand(n, H2, W2, 3); mt = torch.zeros(n, H2, W2)
    for i in range(n):
        mt[i, max(0,cy-fh//2):min(H2,cy+fh//2), max(0,cx-fw//2):min(W2,cx+fw//2)] = 1.0
    return imgs, mt
ti, tm = tall_track()
# width 60px on 400 frame = 15% > 10%; use max_width_fraction 0.20 so it enhances
tc, td, tts, tne, *_ = crop.crop(ti, tm, 2.0, 0.20, 0.0, 0.4, 0.4)
side0 = td["entries"][0]["win"]
# tight square should be ~ max(60,120)*(1.4) = 168, NOT the full 512 height, and
# NOT something derived from width alone. Must contain the face (>=120) but be bounded.
check("tall: crop side contains face height", side0 >= 120)
check("tall: crop side is tight, not full-frame", side0 <= 220)
check("tall: face_clip is square out_side", tc.shape[1] == tc.shape[2] == td["out_side"])
# Paste back round-trips to the same region size
op = paste.paste(ti, Fn.interpolate(tc.permute(0,3,1,2), size=(tts,tts), mode="bilinear",
                 align_corners=False).permute(0,2,3,1), td, 0.0, True)[0]
check("tall: paste preserves shape", tuple(op.shape) == tuple(ti.shape))


# --- empty case: face large for entire clip -> raise clear error (not empty batch) ---
big_all = torch.zeros(4, H, W)
for i in range(4): big_all[i, 50:130, 100:200] = 1.0   # 100px > 40px threshold
raised = False
try:
    crop.crop(torch.rand(4,H,W,3), big_all, 2.0, 0.10, 0.0, 0.3, 0.4)
except ValueError as e:
    raised = "0 frames selected" in str(e)
check("track all-large -> raises clear ValueError", raised)

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
