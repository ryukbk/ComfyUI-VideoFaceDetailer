"""Face Track Crop & Gate's whole_clip crop planning: one size per run, the crop
held still while the face moves inside it, moved only as far as needed, eased
ahead of time, and blind to a mask that drops out for a few frames."""
import numpy as np
import torch

from nodes import FaceTrackCropAndGate, plan_track_crops

W, H = 640, 360


def track(centres, size=24, dropout=()):
    """Frames with a square face of `size` px at each (x, y) centre; on frames
    in `dropout` only the face's left half is in the mask."""
    n = len(centres)
    mask = torch.zeros(n, H, W)
    for i, (x, y) in enumerate(centres):
        x0, y0 = int(x - size / 2), int(y - size / 2)
        x1 = x0 + (size // 2 if i in dropout else size)
        mask[i, y0:y0 + size, x0:x1] = 1.0
    return torch.rand(n, H, W, 3), mask


def crop(images, mask, planning, **kw):
    kw.setdefault("hold", 0.5)
    return FaceTrackCropAndGate().crop(
        images, mask, 2.0, "width", 20.0, 0.0, 1.5, 1.0, 0.5, 0.0, 0.0, "ltx", True, 256,
        crop_planning=planning, plan_hold=kw["hold"], plan_ease_frames=kw.get("ease", 12),
        plan_window=kw.get("window", 9))


def boxes_of(data):
    return [(e["x0"], e["y0"], e["win"]) for e in data["entries"] if e["present"]]


def test_a_face_wobbling_inside_the_crop_does_not_move_it():
    centres = [(320 + (3 if i % 2 else -3), 180 + (2 if i % 3 else -2)) for i in range(40)]
    images, mask = track(centres)
    planned = boxes_of(crop(images, mask, "whole_clip")[1])
    followed = boxes_of(crop(images, mask, "follow")[1])
    assert len({b for b in planned}) == 1                       # one box for the whole run
    assert len({b[:2] for b in followed}) > 1                   # follow mode wobbles with it


def test_a_moving_face_is_followed_only_as_far_as_needed_and_stays_inside():
    centres = [(100 + 8 * i, 180) for i in range(50)]          # 400 px left to right
    images, mask = track(centres)
    data = crop(images, mask, "whole_clip")[1]
    xs = [b[0] for b in boxes_of(data)]
    side = boxes_of(data)[0][2]
    assert len({b[2] for b in boxes_of(data)}) == 1             # one size
    assert all(b >= a for a, b in zip(xs, xs[1:]))               # never moves back
    for (x, _y), (x0, _y0, s) in zip(centres, boxes_of(data)):
        assert x0 + 0 <= x - 12 and x + 12 <= x0 + s             # the face is always inside
    assert xs[-1] - xs[0] < 400                                  # less travel than the face


def test_the_crop_eases_in_before_the_face_moves():
    centres = [(200, 180)] * 20 + [(200 + 10 * i, 180) for i in range(1, 21)]
    images, mask = track(centres)
    eased = [b[0] for b in boxes_of(crop(images, mask, "whole_clip", ease=12)[1])]
    abrupt = [b[0] for b in boxes_of(crop(images, mask, "whole_clip", ease=0)[1])]
    first_move = lambda xs: next(i for i in range(1, len(xs)) if xs[i] != xs[0])
    assert first_move(eased) < first_move(abrupt)                # starts ahead of time
    assert max(np.abs(np.diff(eased))) <= max(np.abs(np.diff(abrupt)))


def test_a_mask_that_drops_out_for_a_few_frames_is_ignored():
    centres = [(320, 180)] * 30
    images, mask = track(centres, dropout=range(12, 15))         # half the face gone for 3 frames
    planned = boxes_of(crop(images, mask, "whole_clip")[1])
    assert len(set(planned)) == 1


def test_a_cut_starts_a_new_shot():
    plan, report = plan_track_crops(list(range(20)), {i: ((100, 100, 123, 123, 112, 112) if i < 10 else
                                                         (500, 200, 523, 223, 512, 212)) for i in range(20)},
                                    [60] * 20, W, H, 1.5)
    assert plan[0] != plan[19] and len(report) == 2
    assert plan[9][0] < 200 < plan[10][0]                        # one jump, at the cut


def test_follow_is_unchanged_by_default():
    images, mask = track([(320 + i, 180) for i in range(10)])
    default = FaceTrackCropAndGate().crop(images, mask, 2.0, "width", 20.0, 0.0, 1.5, 1.0, 0.5, 0.0,
                                          0.0, "ltx", True, 256)[1]
    assert boxes_of(default) == boxes_of(crop(images, mask, "follow")[1])


def growing_face(n=60, start=20, end=60):
    """A face growing from `start` to `end` px, centred."""
    mask = torch.zeros(n, H, W)
    sizes = np.linspace(start, end, n).round().astype(int)
    for i, sz in enumerate(sizes):
        mask[i, 180 - sz // 2:180 - sz // 2 + sz, 320 - sz // 2:320 - sz // 2 + sz] = 1.0
    return torch.rand(n, H, W, 3), mask, sizes


def plan_size(images, mask, size_mode, rate=1.02, ease=12):
    return FaceTrackCropAndGate().crop(
        images, mask, 2.0, "width", 20.0, 0.0, 1.5, 1.0, 5.0, 0.0, 0.0, "ltx", True, 256,
        crop_planning="whole_clip", plan_hold=0.5, plan_ease_frames=ease, plan_window=9,
        plan_size=size_mode, plan_max_zoom_rate=rate)[1]


def test_planned_size_follows_a_growing_face_and_always_holds_it():
    images, mask, sizes = growing_face()
    fixed = boxes_of(plan_size(images, mask, "fixed"))
    follow = boxes_of(plan_size(images, mask, "follow"))
    assert len({b[2] for b in fixed}) == 1
    sides = [b[2] for b in follow]
    assert sides[0] < sides[-1] and all(b >= a for a, b in zip(sides, sides[1:]))   # grows, never back
    for (x0, y0, s), sz in zip(follow, sizes):
        assert s >= sz * (1 + 1.5 * 0.5) - 1                     # keeps half the padding
        assert x0 <= 320 - sz // 2 and 320 - sz // 2 + sz <= x0 + s    # the face is inside


def test_planned_size_holds_still_through_size_jitter():
    mask = torch.zeros(40, H, W)
    for i in range(40):
        sz = 30 + (2 if i % 2 else -2)                          # +-2 px of mask noise
        mask[i, 180 - sz // 2:180 - sz // 2 + sz, 320 - sz // 2:320 - sz // 2 + sz] = 1.0
    follow = boxes_of(plan_size(torch.rand(40, H, W, 3), mask, "follow"))
    assert len({b[2] for b in follow}) == 1


def test_the_zoom_rate_is_capped():
    images, mask, _ = growing_face(n=30, start=20, end=80)
    sides = np.array([b[2] for b in boxes_of(plan_size(images, mask, "follow", rate=1.03))], dtype=float)
    assert (sides[1:] / sides[:-1]).max() <= 1.03 + 2 / sides.min()      # 3% a frame, give or take rounding
