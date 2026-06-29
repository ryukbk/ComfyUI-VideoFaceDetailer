import os
import sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nodes import FaceSizeGateMask

H, W = 200, 400          # frame: 400 wide -> 10% = 40px
g = FaceSizeGateMask()

def mask_with_box(w_px, h_px=30, x=0, y=0):
    m = torch.zeros(H, W)
    m[y:y+h_px, x:x+w_px] = 1.0
    return m

# Frame 0: small face 20px (<40) -> keep
# Frame 1: big face 80px (>=40) -> drop
# Frame 2: exactly 40px -> drop (not < threshold)
# Frame 3: tiny 39px -> keep
# Frame 4: empty -> drop
masks = torch.stack([
    mask_with_box(20),
    mask_with_box(80),
    mask_with_box(40),
    mask_with_box(39),
    torch.zeros(H, W),
])

out, kept, fw = g.gate(masks, 0.10, "bbox_width")
nz = [bool((out[i] > 0.5).any()) for i in range(5)]
print("frame_width:", fw, "(expect 400)")
print("kept_count:", kept, "(expect 2)")
print("per-frame kept(nonzero):", nz, "(expect [True,False,False,True,False])")

ok = (fw == 400 and kept == 2 and nz == [True, False, False, True, False])

# Second check: a 50px face with a 5%(20px) threshold -> drop; with 20%(80px) -> keep
out2, k2, _ = g.gate(mask_with_box(50).unsqueeze(0), 0.05, "bbox_width")
out3, k3, _ = g.gate(mask_with_box(50).unsqueeze(0), 0.20, "bbox_width")
print("50px @5%:", k3 if False else k2, "(expect 0)", "| 50px @20%:", k3, "(expect 1)")
ok = ok and k2 == 0 and k3 == 1

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
