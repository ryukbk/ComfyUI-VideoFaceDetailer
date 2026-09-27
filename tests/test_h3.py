"""FaceTrackCropAndGate + FaceTrackPasteBack on the MiniMax H3 path: clip-length
grids, the track_data contract, graceful no-ops, and the audio slice."""
import pytest
import torch

from nodes import (FaceTrackAudioSlice, FaceTrackCropAndGate, FaceTrackPasteBack,
                   _next_valid_clip_len)

W, H, N = 400, 200, 3


def small_face_clip():
    """Three frames, each with a 20px face (5% of width) — well under the 10% gate."""
    imgs = torch.rand(N, H, W, 3)
    mask = torch.zeros(N, H, W)
    for i in range(N):
        mask[i, 90:110, 190:210] = 1.0
    return imgs, mask


def large_face_clip(imgs):
    """Same frames, but a 200px face (50% of width) — always above the gate."""
    mask = torch.zeros(N, H, W)
    for i in range(N):
        mask[i, 40:160, 100:300] = 1.0
    return mask


def crop(imgs, mask, **kw):
    """Every gate parameter is pinned, none inherited.

    These tests assert on the exact window carried in track_data, so they must state it:
    leaving threshold_type or min_threshold_percent to the node's defaults makes the
    suite fail whenever a default is retuned, which says nothing about the contract.
    """
    kw.setdefault("upscale_ratio", 2.0)
    kw.setdefault("threshold_type", "width")
    kw.setdefault("max_threshold_percent", 10.0)
    kw.setdefault("min_threshold_percent", 0.0)
    kw.setdefault("hysteresis_percent", 2.0)
    kw.setdefault("padding", 0.3)
    kw.setdefault("smooth_alpha", 1.0)
    return FaceTrackCropAndGate().crop(imgs, mask, **kw)


# ── clip-length grids ────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,expected", [(1, 1), (2, 9), (9, 9), (10, 17)])
def test_ltx_pads_to_8n_plus_1(n, expected):
    assert _next_valid_clip_len(n, "ltx") == expected


@pytest.mark.parametrize("n,expected", [(1, 5), (5, 5), (6, 22), (22, 22), (23, 39)])
def test_h3_pads_to_17k_plus_5(n, expected):
    assert _next_valid_clip_len(n, "minimax_h3") == expected


# ── the track_data contract paste-back and H3FaceRefine depend on ────────────

def test_clip_is_padded_onto_the_h3_grid():
    imgs, mask = small_face_clip()
    face_clip, data, _tgt, n_real, *_ = crop(imgs, mask, resampler="minimax_h3")
    clip_len = face_clip.shape[0]

    assert clip_len == _next_valid_clip_len(n_real, "minimax_h3")
    assert (clip_len - 5) % 17 == 0
    assert data["resampler"] == "minimax_h3"
    assert data["clip_length"] == clip_len and data["ltx_length"] == clip_len
    # 1:1 with the clip, so paste-back can index entries by frame position.
    assert len(data["entries"]) == clip_len


def test_entries_carry_what_the_downstream_nodes_read():
    imgs, mask = small_face_clip()
    _clip, data, _tgt, n_real, *_ = crop(imgs, mask, resampler="minimax_h3")
    present = [e for e in data["entries"] if e.get("present")]

    assert len(present) == n_real
    assert all(e["face_px"] > 0 for e in present)
    # H3FaceRefine ramps per-frame denoise between the gate's thresholds using these.
    assert data["threshold_type"] == "width"
    assert data["max_threshold_frac"] == pytest.approx(0.10)
    assert data["min_threshold_frac"] == 0.0
    assert all(e["measure_frac"] > 0 for e in present)


def test_resampler_defaults_to_h3_and_ltx_is_honoured():
    imgs, mask = small_face_clip()
    _c, default_data, _t, _n, _r, default_fc, *_ = crop(imgs, mask)
    assert default_data["resampler"] == "minimax_h3"
    assert (default_data["clip_length"] - 5) % 17 == 0

    _c, ltx_data, _t, _n, _r, ltx_fc, *_ = crop(imgs, mask, resampler="ltx")
    assert ltx_data["resampler"] == "ltx"
    assert (ltx_data["clip_length"] - 1) % 8 == 0

    # frame_count is what drives the resampler's `length`, so it must be the PADDED length.
    assert default_fc == default_data["clip_length"]
    assert ltx_fc == ltx_data["clip_length"]


def test_enhanced_flag_and_report():
    imgs, mask = small_face_clip()
    _c, _d, _t, n_real, _r, _fc, enhanced, report = crop(imgs, mask, resampler="minimax_h3")
    assert enhanced is True and n_real > 0
    # The report is how the user picks thresholds, so it has to quote all three units.
    assert all(k in report for k in ("width", "height", "area"))
    assert "min–max" in report and "enhanced" in report


# ── graceful no-ops: nothing qualifies ───────────────────────────────────────

def test_a_face_above_the_gate_throughout_is_a_no_op():
    imgs, _ = small_face_clip()
    clip, data, target_size, n_real, n_runs, frame_count, enhanced, _rep = crop(
        imgs, large_face_clip(imgs), hysteresis_percent=0.0, resampler="minimax_h3")

    assert n_real == 0 and n_runs == 0
    assert enhanced is False, "the If/Else Switch uses this to skip the whole enhance branch"
    assert target_size >= 8, "0 would fail ImageResizeKJv2 ('must be > 0')"
    assert frame_count == clip.shape[0] >= 5
    assert (clip.shape[0] - 5) % 17 == 0
    assert all(not e.get("present") for e in data["entries"])


def test_an_empty_enable_window_is_a_no_op_not_a_crash():
    # min_threshold == max_threshold: the reported bug config.
    imgs, mask = small_face_clip()
    _c, _d, _t, n_real, _r, _fc, enhanced, _rep = crop(
        imgs, mask, hysteresis_percent=0.0, min_threshold_percent=10.0,
        resampler="minimax_h3")
    assert n_real == 0 and enhanced is False


def test_paste_back_on_a_no_op_clip_returns_the_original_untouched():
    imgs, _ = small_face_clip()
    clip, data, target_size, *_ = crop(imgs, large_face_clip(imgs),
                                       hysteresis_percent=0.0, resampler="minimax_h3")
    processed = torch.rand(clip.shape[0], target_size, target_size, 3)
    (out,) = FaceTrackPasteBack().paste(imgs, processed, data, feather=0.15,
                                        blend_mode="mask", only_present_frames=True,
                                        colour_match=0.0)
    assert torch.equal(out, imgs)


# ── paste-back ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("blend_mode", ["mask", "rectangle"])
def test_paste_back_with_colour_match(blend_mode):
    imgs, mask = small_face_clip()
    clip, data, target_size, *_ = crop(imgs, mask, resampler="minimax_h3")
    processed = torch.rand(clip.shape[0], target_size, target_size, 3)
    (out,) = FaceTrackPasteBack().paste(imgs, processed, data, feather=0.15,
                                        blend_mode=blend_mode,
                                        only_present_frames=True, colour_match=1.0)
    assert tuple(out.shape) == (N, H, W, 3)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


# ── audio, reindexed onto the gated clip ─────────────────────────────────────

def test_audio_is_sliced_to_the_clip_frames():
    imgs, mask = small_face_clip()
    _c, data, *_ = crop(imgs, mask, resampler="minimax_h3")
    sr = 16000
    audio = {"waveform": torch.zeros(1, 2, sr * 3), "sample_rate": sr}   # 3s stereo

    out, _report = FaceTrackAudioSlice().slice(audio, data, source_fps=24.0, target_fps=24.0)

    assert out["waveform"].shape[-1] == len(data["entries"]) * round(sr / 24.0)
    assert out["sample_rate"] == sr
    assert out["waveform"].shape[:2] == (1, 2)


def test_audio_slice_tolerates_a_missing_waveform():
    imgs, mask = small_face_clip()
    _c, data, *_ = crop(imgs, mask, resampler="minimax_h3")
    out, _report = FaceTrackAudioSlice().slice({"waveform": None, "sample_rate": 0}, data)
    assert out is not None


# ── the per-run H3 workflow depends on slicing audio PER RUN ──────────────────

def multi_run_clip():
    """small, small | big, big, big | small, small, small — two enhanced runs."""
    widths = [24, 28, 60, 100, 60, 24, 20, 22]
    imgs = torch.rand(len(widths), H, W, 3)
    mask = torch.zeros(len(widths), H, W)
    for i, width in enumerate(widths):
        half = width // 2
        mask[i, 100 - half:100 + half, 100 - half:100 + half] = 1.0
    return imgs, mask


def ramped_audio(frames, sr=16000, fps=24.0):
    """A waveform whose every source frame carries its own constant, so a slice can
    be traced back to the frames it was taken from."""
    win = round(sr / fps)
    wave = torch.zeros(1, 1, win * (frames + 4))
    for f in range(frames):
        wave[0, 0, f * win:(f + 1) * win] = f + 1
    return {"waveform": wave, "sample_rate": sr}, win


def source_frames_in(sliced, win):
    values = sliced["waveform"][0, 0]
    seen = []
    for k in range(0, values.shape[0], win):
        v = int(values[k:k + win].max().item())
        if v and (not seen or seen[-1] != v):
            seen.append(v - 1)
    return seen


def test_each_run_slices_its_own_audio():
    """Why face_enhance_h3_track_perrun needs a FaceTrackAudioSlice per branch: one
    clip-wide slice spans every run, so it would drift against all but the first."""
    from nodes import FaceTrackSelectRun

    imgs, mask = multi_run_clip()
    clip, data, _tgt, _n_real, n_runs, *_ = crop(
        imgs, mask, hysteresis_percent=0.0, padding=1.5, resampler="minimax_h3")
    assert n_runs == 2, "the fixture must actually split into two runs"

    audio, win = ramped_audio(imgs.shape[0])
    whole, _ = FaceTrackAudioSlice().slice(audio, data, source_fps=24.0, target_fps=24.0)
    assert source_frames_in(whole, win) == [0, 1, 5, 6, 7], "clip-wide slice spans both runs"

    for run in range(n_runs):
        _run_clip, run_data, *_ = FaceTrackSelectRun().select(clip, data, run)
        sliced, _ = FaceTrackAudioSlice().slice(audio, run_data,
                                                source_fps=24.0, target_fps=24.0)
        expected = [e["frame"] for e in run_data["entries"] if e["present"]]
        assert source_frames_in(sliced, win) == expected
