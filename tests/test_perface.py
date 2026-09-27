"""FaceCropAndGate / FacePasteBack: one crop per FACE, pasted back into its own frame."""
import numpy as np
import pytest
import torch

from nodes import FaceCropAndGate, FacePasteBack, _connected_components

B, H, W, C = 3, 200, 400, 3      # 400 wide -> 10% = 40px


def two_face_video():
    """frame 0: a 20px face (kept) and a 60px face (skipped); frame 1: 30px; frame 2: none."""
    imgs = torch.rand(B, H, W, C)
    masks = torch.zeros(B, H, W)
    masks[0, 30:50, 20:40] = 1.0        # w=20 -> keep
    masks[0, 30:70, 200:260] = 1.0      # w=60 -> skip
    masks[1, 80:110, 100:130] = 1.0     # w=30 -> keep
    return imgs, masks


def _crop(imgs, masks, fraction=0.10):
    return FaceCropAndGate().crop(imgs, masks, fraction, "bbox_width", 0.3, 256, 8)


def test_connected_components_separates_two_faces():
    m = np.zeros((100, 200), bool)
    m[10:20, 10:30] = True              # face A (w=20)
    m[50:90, 120:160] = True            # face B (w=40)
    assert len(_connected_components(m)) == 2


def test_only_small_faces_are_cropped():
    imgs, masks = two_face_video()
    face_crops, face_data, n = _crop(imgs, masks)

    assert n == 2
    assert tuple(face_crops.shape) == (2, 256, 256, 3)
    # Each entry knows which frame it came from, which is what makes multi-face
    # paste-back land in the right place.
    assert sorted(e["frame"] for e in face_data["entries"]) == [0, 1]


def test_paste_back_touches_only_the_frames_that_had_a_face():
    imgs, masks = two_face_video()
    face_crops, face_data, _ = _crop(imgs, masks)
    out = FacePasteBack().paste(imgs, face_crops.clone(), face_data, 0.0)[0]

    assert tuple(out.shape) == (B, H, W, C)
    assert torch.equal(out[2], imgs[2]), "frame 2 had no face and must be untouched"
    assert not torch.equal(out[0], imgs[0]), "frame 0 had a face and must change"


def test_paste_back_tolerates_a_short_processed_batch():
    # A count mismatch is a user wiring error; it must warn and paste what it can
    # rather than raising in the middle of a render.
    imgs, masks = two_face_video()
    face_crops, face_data, _ = _crop(imgs, masks)
    out = FacePasteBack().paste(imgs, face_crops[:1], face_data, 0.1)[0]
    assert tuple(out.shape) == (B, H, W, C)


def test_all_faces_too_big_raises_rather_than_emitting_an_empty_batch():
    # An empty batch would crash downstream on torch.stack / Resize instead of
    # telling the user their threshold is wrong.
    big = torch.zeros(2, H, W)
    big[0, 0:80, 0:120] = 1.0           # w=120, well over the 40px gate
    with pytest.raises(ValueError, match="0 faces selected"):
        FaceCropAndGate().crop(torch.rand(2, H, W, C), big, 0.10, "bbox_width", 0.3, 256, 8)
