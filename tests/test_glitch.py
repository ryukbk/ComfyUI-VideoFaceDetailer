"""Stray mask spans: a tracked mask that jumps away and comes back within
glitch_frames is replaced by the face where it would be; a jump that stays
(a cut, or the face really moving) is kept."""
import torch

from nodes import FaceTrackCropAndGate, find_mask_glitches, repair_mask_glitches

W, H = 640, 360


def face_at(n, where):
    """n frames, a 24 px face at where(i) = (x, y) on frame i."""
    mask = torch.zeros(n, H, W)
    for i in range(n):
        x, y = where(i)
        mask[i, y - 12:y + 12, x - 12:x + 12] = 1.0
    return mask


def box(x, y, size=24):
    return (x - size // 2, y - size // 2, x + size // 2 - 1, y + size // 2 - 1, float(x), float(y))


def test_a_jump_that_comes_back_is_a_glitch_and_one_that_stays_is_not():
    away = [box(300, 180)] * 10 + [box(550, 60)] * 3 + [box(304, 180)] * 10
    assert find_mask_glitches(away, max_frames=6) == [(10, 13, 9, 13)]
    stays = [box(300, 180)] * 10 + [box(550, 60)] * 13            # a cut
    assert find_mask_glitches(stays, max_frames=6) == []
    assert find_mask_glitches(away, max_frames=2) == []           # longer than allowed: kept
    assert find_mask_glitches(away, max_frames=0) == []           # off


def test_a_face_moving_steadily_is_never_a_glitch():
    moving = [box(100 + 6 * i, 180) for i in range(60)]
    assert find_mask_glitches(moving, max_frames=6) == []


def test_the_stray_frames_get_the_face_moved_along_its_path():
    mask = face_at(12, lambda i: (550, 60) if i in (5, 6) else (200 + 4 * i, 180))
    boxes = [box(550, 60) if i in (5, 6) else box(200 + 4 * i, 180) for i in range(12)]
    spans = find_mask_glitches(boxes, max_frames=6)
    fixed_boxes, fixed = repair_mask_glitches(boxes, mask, spans)
    assert fixed is not mask and mask[5, 60, 550] == 1.0          # the input is not changed
    assert fixed[5, 60, 550] == 0.0                               # the stray mask is gone
    assert fixed[5, 180, 220] == 1.0 and fixed[6, 180, 224] == 1.0   # the face, on its path
    assert fixed_boxes[5][4:] == (220.0, 180.0)


def test_the_crop_and_its_masks_ignore_a_stray_span():
    mask = face_at(30, lambda i: (560, 60) if 12 <= i < 15 else (320, 180))
    images = torch.rand(30, H, W, 3)
    args = (images, mask, 2.0, "width", 20.0, 0.0, 1.5, 1.0, 0.5, 0.0, 0.0, "ltx", True, 256)
    kept = FaceTrackCropAndGate().crop(*args, crop_planning="whole_clip", plan_window=1,
                                       glitch_frames=0, plan_return_frames=0)[1]
    cleaned_out = FaceTrackCropAndGate().crop(*args, crop_planning="whole_clip", plan_window=1,
                                              glitch_frames=6)
    cleaned = cleaned_out[1]
    present = lambda d: [e for e in d["entries"] if e["present"]]
    assert len({(e["x0"], e["y0"]) for e in present(kept)}) > 1           # without: it jumps
    assert len({(e["x0"], e["y0"], e["win"]) for e in present(cleaned)}) == 1   # with: still
    assert all(e["cmask"].sum() > 0 for e in present(cleaned))             # the face in every crop
    assert "stray mask ignored on frames 12-14" in cleaned_out[7]


def test_interpolate_blends_size_and_shape_between_the_good_frames():
    mask = torch.zeros(10, H, W)
    for i in range(10):
        size = 20 if i < 4 else 40                               # the face grows across the span
        x = 200 + 4 * i
        mask[i, 180 - size // 2:180 + size // 2, x - size // 2:x + size // 2] = 1.0
    mask[4:6] = 0
    mask[4:6, 50:74, 550:574] = 1.0                              # the stray mask, frames 4-5
    boxes = [None] * 10
    from nodes import _frame_face_box
    boxes = [_frame_face_box((mask[i] > 0.5).numpy()) for i in range(10)]
    spans = find_mask_glitches(boxes, max_frames=6, distance=1.0)
    assert spans == [(4, 6, 3, 6)]
    nb, nearest = repair_mask_glitches(boxes, mask, spans, "move_nearest")
    ib, interp = repair_mask_glitches(boxes, mask, spans, "interpolate")
    side = lambda b: max(b[2] - b[0] + 1, b[3] - b[1] + 1)
    assert [side(nb[i]) for i in (4, 5)] == [20, 40]             # nearest: one or the other
    assert 20 < side(ib[4]) < side(ib[5]) < 40                   # interpolate: in between, growing
    assert interp[4, 60, 560] == 0 and interp[5, 60, 560] == 0   # the stray mask is gone
    assert abs(ib[4][4] - 216) <= 2 and abs(ib[5][4] - 220) <= 2   # the face's path, kept moving


def test_interpolate_keeps_the_speed_at_both_ends():
    from nodes import _interpolated_face
    boxes = [box(100 + 10 * i, 180) for i in range(20)]           # steady 10 px a frame
    centres = [_interpolated_face(boxes, 7, 12, i, set(range(8, 12)))[0][0] for i in range(8, 12)]
    assert all(abs(c - (100 + 10 * i)) < 1e-6 for c, i in zip(centres, range(8, 12)))


def test_the_crop_holds_still_with_interpolate_too():
    mask = face_at(30, lambda i: (560, 60) if 12 <= i < 15 else (320, 180))
    args = (torch.rand(30, H, W, 3), mask, 2.0, "width", 20.0, 0.0, 1.5, 1.0, 0.5, 0.0, 0.0, "ltx", True, 256)
    out = FaceTrackCropAndGate().crop(*args, crop_planning="whole_clip", plan_window=1,
                                      glitch_frames=6, glitch_fill="interpolate")
    present = [e for e in out[1]["entries"] if e["present"]]
    assert len({(e["x0"], e["y0"], e["win"]) for e in present}) == 1
    assert all(e["cmask"].sum() > 0 for e in present)
    assert "(interpolate)" in out[7]


def test_the_preview_track_is_what_sam3_track_preview_reads():
    mask = face_at(30, lambda i: (560, 60) if 12 <= i < 15 else (320, 180))
    out = FaceTrackCropAndGate().crop(torch.rand(30, H, W, 3), mask, 2.0, "width", 20.0, 0.0, 1.5, 1.0,
                                      0.5, 0.0, 0.0, "ltx", True, 256, crop_planning="whole_clip",
                                      glitch_frames=6)
    track = out[8]
    packed = track["packed_masks"]
    assert packed.dtype == torch.uint8 and packed.shape == (30, 3, H, W // 8)
    assert track["n_frames"] == 30 and track["orig_size"] == (H, W) and track["scores"] == [1.0] * 3
    bits = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)      # ComfyUI's unpack_masks
    unpacked = (packed.unsqueeze(-1) & bits).bool().view(30, 3, H, W)
    assert unpacked[13, 0, 180, 320] and not unpacked[13, 0, 60, 560]          # the face used, not the stray
    assert unpacked[13, 2, 60, 560] and not unpacked[20, 2].any()               # the stray, only where replaced
    entry = next(e for e in out[1]["entries"] if e["frame"] == 20)
    assert unpacked[20, 1, entry["y0"], entry["x0"] + 5]                        # the crop box's edge
    assert not unpacked[20, 1, entry["y0"] + entry["win"] // 2, entry["x0"] + entry["win"] // 2]   # hollow
