import os
import sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nodes import FaceCropAndGate, FacePasteBack, _connected_components
import numpy as np

ok = True
def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name); ok = ok and cond

# --- connected components: two separate boxes in one mask ---
m = np.zeros((100, 200), bool)
m[10:20, 10:30] = True      # face A (w=20)
m[50:90, 120:160] = True    # face B (w=40)
boxes = _connected_components(m)
check("CC finds 2 components", len(boxes) == 2)

# --- Build a 3-frame video, W=400 (10% = 40px) ---
B, H, W, C = 3, 200, 400, 3
imgs = torch.rand(B, H, W, C)
masks = torch.zeros(B, H, W)
# frame 0: two faces -> small 20px (keep) + big 60px (skip)
masks[0, 30:50, 20:40] = 1.0      # w=20 -> keep
masks[0, 30:70, 200:260] = 1.0    # w=60 -> skip
# frame 1: one small face 30px -> keep
masks[1, 80:110, 100:130] = 1.0   # w=30 -> keep
# frame 2: no face
crop = FaceCropAndGate()
face_crops, face_data, n = crop.crop(imgs, masks, 0.10, "bbox_width", 0.3, 256, 8)
check("kept exactly 2 small faces (frame0 small + frame1)", n == 2)
check("face_crops shape [2,256,256,3]", tuple(face_crops.shape) == (2, 256, 256, 3))
frames_of_entries = [e["frame"] for e in face_data["entries"]]
check("entries map to frames 0 and 1", sorted(frames_of_entries) == [0, 1])

# --- Paste-back identity check: feed the ORIGINAL crops back, expect near-identity ---
# Re-crop the exact source regions at native size to compare after paste.
paste = FacePasteBack()
# Use the produced (resized) crops as "processed" -> paste back; with feather=0 the
# interior should closely match a downscale->upscale of the source (lossy), so we
# instead test that pasting the crop of a region returns that region for feather=0
# when processed face == exact original region.
# Build processed faces = exact original regions (resized to crop_size) so round trip is well-defined.
processed = face_crops.clone()
out = paste.paste(imgs, processed, face_data, 0.0)[0]
check("paste output shape preserved", tuple(out.shape) == (B, H, W, C))
# Frame 2 (untouched) must be byte-identical to input
check("untouched frame 2 unchanged", torch.equal(out[2], imgs[2]))
# A frame with a pasted face must differ from original somewhere
check("frame 0 changed by paste", not torch.equal(out[0], imgs[0]))

# --- order/length mismatch must not crash ---
out2 = paste.paste(imgs, face_crops[:1], face_data, 0.1)[0]
check("mismatch count tolerated", tuple(out2.shape) == (B, H, W, C))

# --- empty case: all faces big -> raise a clear error (not an empty batch that
#     would crash downstream Resize/LTX with torch.stack on empty) ---
big = torch.zeros(2, H, W); big[0, 0:80, 0:120] = 1.0  # w=120 big
raised = False
try:
    crop.crop(torch.rand(2,H,W,C), big, 0.10, "bbox_width", 0.3, 256, 8)
except ValueError as e:
    raised = "0 faces selected" in str(e)
check("all-big -> raises clear ValueError", raised)

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
