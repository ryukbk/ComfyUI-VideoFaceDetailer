"""H3FaceRefine's video noise mask: the face region (mask_region='face') and the
reduction onto MiniMax H3's latent grid (1, 4, 4, 4, 4 frames per 17-frame
chunk, the last 3 latents dropped; the DiT reads 2x2 latent patches)."""
import sys
import types

import pytest
import torch

import nodes
from nodes import (H3FaceRefine, face_region_masks, h3_latent_groups, latent_mask_preview,
                   latent_noise_mask)


def video_latent_t(frames):
    """Core H3's latent count for a 17k+5 clip (comfy_extras/nodes_minimax_h3.py)."""
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2


# ── the frame groups ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("frames", [5, 22, 39, 56, 124, 362])
def test_groups_match_h3s_latent_count_and_cover_every_frame(frames):
    groups = h3_latent_groups(frames)
    assert len(groups) == video_latent_t(frames)
    assert groups[0] == (0, 1)                                   # the first latent is one frame
    assert [b for _a, b in groups[:-1]] == [a for a, _b in groups[1:]]     # contiguous
    assert groups[-1][1] == frames


def test_groups_follow_the_17_frame_chunks():
    assert h3_latent_groups(22) == [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 18), (18, 22)]


def test_groups_read_the_vaes_chunking():
    inner = types.SimpleNamespace(clip_length=9, vae_ratio_t=4, token_drop=0)
    vae = types.SimpleNamespace(first_stage_model=inner)
    assert h3_latent_groups(18, vae) == [(0, 1), (1, 5), (5, 9), (9, 10), (10, 14), (14, 18)]


# ── the reduction ────────────────────────────────────────────────────────────

def test_a_cell_is_resampled_if_any_pixel_or_frame_it_covers_is():
    frames, h, w = 22, 64, 64
    region = torch.zeros(frames, h, w)
    region[3, 20, 40] = 1.0                    # one pixel on one frame (latent 1: frames 1-4)
    mask = latent_noise_mask([1.0] * frames, h3_latent_groups(frames), 4, 4, region)
    assert mask.shape == (7, 4, 4)
    assert mask[1, 1, 2] == 1.0                # its 16 px cell, on its latent frame
    assert mask.sum() == 1.0                   # nothing else


def test_the_strength_of_a_latent_frame_is_its_strongest_frames():
    groups = h3_latent_groups(22)
    strength = [0.2] * 22
    strength[7] = 0.9                          # in latent 2 (frames 5-8)
    mask = latent_noise_mask(strength, groups, 2, 2)
    assert torch.allclose(mask[:, 0, 0], torch.tensor([0.2, 0.2, 0.9, 0.2, 0.2, 0.2, 0.2]))


def test_the_preview_shows_2x2_patches_on_their_frames():
    groups = h3_latent_groups(22)
    lmask = torch.zeros(7, 4, 4)
    lmask[2, 1, 2] = 0.5
    preview = latent_mask_preview(lmask, groups, 64, 64, 22)
    assert preview.shape == (22, 64, 64)
    assert preview[5:9, 0:32, 32:64].eq(0.5).all()             # the whole 2x2 patch, frames 5-8
    assert preview[:5].eq(0).all() and preview[9:].eq(0).all()


# ── the face region ──────────────────────────────────────────────────────────

def entries_with(mask_hw, n, pads=0):
    return ([{"present": True, "cmask": mask_hw} for _ in range(n)]
            + [{"present": False, "cmask": None} for _ in range(pads)])


def test_the_face_mask_is_grown_feathered_and_carried_onto_pad_frames():
    cm = torch.zeros(10, 10)
    cm[4:6, 4:6] = 1.0                          # a 2x2 face in a 10x10 crop
    region = face_region_masks(entries_with(cm, 2, pads=1), 100, 100, grow_px=0, feather_px=0)
    assert region.shape == (3, 100, 100)
    assert region[0, 40:60, 40:60].eq(1).all() and region[0].sum() == 400       # resized to the clip
    assert torch.equal(region[2], region[1])                                      # the pad repeats
    grown = face_region_masks(entries_with(cm, 1), 100, 100, grow_px=10, feather_px=0)
    assert grown[0].sum() == 40 * 40
    soft = face_region_masks(entries_with(cm, 1), 100, 100, grow_px=10, feather_px=6)
    assert 0 < soft[0, 50, 27] < 1                                                # a soft edge
    assert soft[0, 50, 50] > 0.999                                                # full inside


def test_no_mask_at_all_resamples_the_whole_frame():
    region = face_region_masks(entries_with(None, 2), 8, 8)
    assert region.eq(1).all()


# ── H3FaceRefine end to end, with stand-ins for ComfyUI ──────────────────────

class NestedTensor:
    def __init__(self, tensors):
        self.tensors = tuple(tensors)

    def unbind(self):
        return self.tensors


class VideoVAE:
    def encode(self, images):
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        return torch.zeros(1, 24, video_latent_t(n), h // 16, w // 16)


@pytest.fixture
def comfy_stub(monkeypatch):
    comfy = types.ModuleType("comfy")
    comfy.nested_tensor = types.SimpleNamespace(NestedTensor=NestedTensor)
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", comfy.nested_tensor)


def refine(mask_region, **kw):
    frames, side = 22, 128
    cm = torch.zeros(32, 32)
    cm[8:16, 8:16] = 1.0                         # the face: the crop's upper-left quarter
    entries = [{"present": True, "cmask": cm, "measure_frac": 0.01} for _ in range(20)] + \
              [{"present": False, "cmask": None} for _ in range(2)]
    track = {"entries": entries, "min_threshold_frac": 0.001, "max_threshold_frac": 0.05,
             "threshold_type": "area"}
    latent = {"samples": NestedTensor((torch.zeros(1, 24, 7, side // 16, side // 16),
                                       torch.zeros(1, 64, 22)))}
    return H3FaceRefine().run(None, latent, torch.rand(frames, side, side, 3), VideoVAE(), track,
                              1.0, 0.35, 1.0, 1, mask_region=mask_region, **kw)


def test_face_mode_resamples_only_the_face_and_keeps_the_rest(comfy_stub):
    _model, latent, report, preview = refine("face", mask_grow=0, mask_feather=0)
    video_mask = latent["noise_mask"].unbind()[0]
    assert video_mask.shape == (1, 24, 7, 8, 8)
    cells = video_mask[0, 0, 0]
    assert cells[2:4, 2:4].gt(0).all()          # the face's latent cells (32-64 px of 128)
    assert cells[4:, :].eq(0).all() and cells[:, 4:].eq(0).all()     # the rest is kept
    assert preview.shape == (22, 128, 128)
    assert "mask: face" in report


def test_crop_mode_is_uniform_over_the_crop(comfy_stub):
    _model, latent, report, preview = refine("crop")
    video_mask = latent["noise_mask"].unbind()[0]
    assert video_mask[0, 0].std(dim=(-2, -1)).max() == 0        # one value per latent frame
    assert video_mask.min() > 0
    assert "mask: whole crop" in report


def test_growing_is_a_square_max_filter_and_fast_at_clip_size():
    import time
    import torch.nn.functional as F
    from nodes import _grow_mask
    torch.manual_seed(0)
    m = (torch.rand(2, 120, 160) > 0.99).float()
    m[:, 40:70, 50:90] = 1.0
    for r in (1, 7, 32):
        grown = F.max_pool2d(m[:, None], 2 * r + 1, stride=1, padding=r)[:, 0]
        shrunk = -F.max_pool2d(-m[:, None], 2 * r + 1, stride=1, padding=r)[:, 0]
        assert torch.equal(_grow_mask(m, r), grown) and torch.equal(_grow_mask(m, -r), shrunk)
    # It froze H3FaceRefine: a 65 x 65 max filter took ~13 s per 768 px frame on CPU.
    cm = torch.zeros(300, 300)
    cm[100:200, 110:190] = 1.0
    start = time.time()
    face_region_masks([{"present": True, "cmask": cm.clone()} for _ in range(20)], 768, 768, 32, 16)
    assert time.time() - start < 5.0
