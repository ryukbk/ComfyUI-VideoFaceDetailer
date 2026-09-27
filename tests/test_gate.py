"""FaceSizeGateMask: keep a frame's mask only while the face is under the threshold."""
import torch

from nodes import FaceSizeGateMask

H, W = 200, 400          # frame: 400 wide -> 10% = 40px


def mask_with_box(w_px, h_px=30, x=0, y=0):
    m = torch.zeros(H, W)
    m[y:y + h_px, x:x + w_px] = 1.0
    return m


def _gate(masks, fraction=0.10):
    return FaceSizeGateMask().gate(masks, fraction, "bbox_width")


def test_gate_keeps_only_faces_under_the_threshold():
    # 20px small -> keep; 80px big -> drop; exactly 40px -> drop (not < threshold);
    # 39px -> keep; empty -> drop.
    masks = torch.stack([
        mask_with_box(20),
        mask_with_box(80),
        mask_with_box(40),
        mask_with_box(39),
        torch.zeros(H, W),
    ])
    out, kept, frame_width = _gate(masks)

    assert frame_width == 400
    assert kept == 2
    nonzero = [bool((out[i] > 0.5).any()) for i in range(5)]
    assert nonzero == [True, False, False, True, False]


def test_gate_threshold_is_a_fraction_of_frame_width():
    # The same 50px face: 5% of 400 is 20px (drop), 20% is 80px (keep).
    _, tight, _ = _gate(mask_with_box(50).unsqueeze(0), 0.05)
    _, loose, _ = _gate(mask_with_box(50).unsqueeze(0), 0.20)
    assert tight == 0
    assert loose == 1
