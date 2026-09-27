"""FaceTrackCropAndGate: the size gate over a zoom, and the crop window it produces."""
import torch
import torch.nn.functional as Fn

from nodes import FaceTrackCropAndGate, FaceTrackPasteBack

W, H = 400, 200          # 400 wide -> 10% = 40px


def crop_on_width(imgs, mask_track, ratio, max_percent, hysteresis_percent,
                  padding, smooth_alpha, **kw):
    """Gate on face WIDTH, which is what every test here measures.

    A named helper rather than positional calls: the node takes fourteen parameters and
    the thresholds are PERCENTS (10.0), not the fractions (0.10) this suite used before
    the percent rework - passing one for the other silently empties the enable window
    instead of failing, so it is worth naming them at the call site.
    """
    return FaceTrackCropAndGate().crop(
        imgs, mask_track,
        upscale_ratio=ratio,
        threshold_type="width",
        max_threshold_percent=max_percent,
        hysteresis_percent=hysteresis_percent,
        padding=padding,
        smooth_alpha=smooth_alpha,
        min_threshold_percent=kw.pop("min_percent", 0.0),
        resampler=kw.pop("resampler", "ltx"),
        **kw,
    )


def face_track(widths, cx=100, cy=100, w=W, h=H):
    """Frames each holding one square face of the given width; None = no face."""
    imgs = torch.rand(len(widths), h, w, 3)
    mask = torch.zeros(len(widths), h, w)
    for i, width in enumerate(widths):
        if width is None:
            continue
        half = width // 2
        mask[i, max(0, cy - half):min(h, cy + half),
             max(0, cx - half):min(w, cx + half)] = 1.0
    return imgs, mask


def present_frames(data):
    return [e["frame"] for e in data["entries"] if e["present"]]


# ── the zoom scenario: face grows past the gate mid-clip ─────────────────────
ZOOM_WIDTHS = [24, 28, 34, 44, 70, 120]     # crosses 40px at frame 3


def test_zoom_enhances_only_the_small_frames():
    imgs, mask = face_track(ZOOM_WIDTHS)
    _clip, data, _tgt, n_real, *_ = crop_on_width(imgs, mask, 2.0, 10.0, 0.0, 0.3, 0.4)
    assert present_frames(data) == [0, 1, 2]
    assert n_real == 3


def test_zoom_leaves_the_large_frames_byte_identical():
    imgs, mask = face_track(ZOOM_WIDTHS)
    clip, data, tgt, *_ = crop_on_width(imgs, mask, 2.0, 10.0, 0.0, 0.3, 0.4)
    processed = Fn.interpolate(clip.permute(0, 3, 1, 2), size=(tgt, tgt),
                               mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    out = FaceTrackPasteBack().paste(imgs, processed, data, 0.0, "mask", True)[0]

    assert torch.equal(out[4], imgs[4])
    assert torch.equal(out[5], imgs[5])
    assert not torch.equal(out[0], imgs[0]), "frame 0 was enhanced and must change"


def test_zoom_crop_boxes_stay_tight_to_the_small_faces():
    # Sized per frame from that frame's own face, not from the 120px close-up.
    imgs, mask = face_track(ZOOM_WIDTHS)
    _clip, data, *_ = crop_on_width(imgs, mask, 2.0, 10.0, 0.0, 0.3, 0.4)
    assert max(e["win"] for e in data["entries"]) < 70


def test_zoom_out_and_in_again_is_two_runs():
    imgs, mask = face_track([24, 60, 100, 60, 24, 20])   # big in the middle
    _clip, data, _tgt, _n_real, n_runs, *_ = crop_on_width(imgs, mask, 2.0, 10.0, 0.0, 0.3, 0.4)
    assert present_frames(data) == [0, 4, 5]
    assert n_runs == 2


def test_hysteresis_stops_the_gate_flickering():
    # 12% on / 3% dead-band: ON below 9% (36px), OFF at/above 15% (60px). The face
    # opens at 30px and then hovers at 39/41px, so it must stay ON throughout.
    imgs, mask = face_track([30, 39, 41, 39, 41, 39])
    _clip, data, _tgt, n_real, *_ = crop_on_width(imgs, mask, 2.0, 12.0, 3.0, 0.3, 0.4)
    assert present_frames(data) == [0, 1, 2, 3, 4, 5]
    assert n_real == 6


def test_frames_with_no_face_are_never_enhanced():
    imgs, mask = face_track([24, None, 28, None, 30])
    _clip, data, *_ = crop_on_width(imgs, mask, 2.0, 10.0, 0.0, 0.3, 0.4)
    assert present_frames(data) == [0, 2, 4]


def test_upscale_ratio_scales_target_size_but_not_the_paste_region():
    # The crop box is where the face IS; the ratio only decides how big the
    # resampler works. Mixing the two would move the paste-back.
    # fixed_target_size=False explicitly: this is the ratio mode's behaviour, and the
    # node now defaults to the fixed mode, where the ratio deliberately has no effect.
    imgs, mask = face_track(ZOOM_WIDTHS)
    _c2, d2, t2, *_ = crop_on_width(imgs, mask, 2.0, 10.0, 0.0, 0.3, 0.4,
                                    fixed_target_size=False)
    _c4, d4, t4, *_ = crop_on_width(imgs, mask, 4.0, 10.0, 0.0, 0.3, 0.4,
                                    fixed_target_size=False)

    assert t4 > t2
    assert d2["entries"][0]["win"] == d4["entries"][0]["win"]
    assert d2["entries"][0]["x0"] == d4["entries"][0]["x0"]


def test_a_tall_face_does_not_bloat_the_crop():
    """Regression: a mask taller than it is wide once produced a full-height crop."""
    h2, w2, fw, fh = 512, 400, 60, 120
    imgs = torch.rand(4, h2, w2, 3)
    mask = torch.zeros(4, h2, w2)
    mask[:, 256 - fh // 2:256 + fh // 2, 200 - fw // 2:200 + fw // 2] = 1.0

    # 60px of 400 is 15%, so the gate has to be opened past it.
    clip, data, tgt, *_ = crop_on_width(imgs, mask, 2.0, 20.0, 0.0, 0.4, 0.4)
    side = data["entries"][0]["win"]

    assert side >= fh, "the crop must contain the face's HEIGHT"
    assert side <= 220, "and must not grow to the full frame height"
    assert clip.shape[1] == clip.shape[2] == data["out_side"], "crops are square"

    processed = Fn.interpolate(clip.permute(0, 3, 1, 2), size=(tgt, tgt),
                               mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    out = FaceTrackPasteBack().paste(imgs, processed, data, 0.0, "mask", True)[0]
    assert tuple(out.shape) == tuple(imgs.shape)


def test_a_face_large_for_the_whole_clip_is_a_no_op_not_an_error():
    """It used to raise. A no-op passthrough is right: the video is still usable,
    and raising mid-render loses the whole queue item."""
    imgs = torch.rand(4, H, W, 3)
    big = torch.zeros(4, H, W)
    big[:, 50:130, 100:200] = 1.0       # 100px, well over the 40px gate

    clip, data, target_size, n_real, n_runs, frame_count, enhanced, _report = \
        crop_on_width(imgs, big, 2.0, 10.0, 0.0, 0.3, 0.4)

    assert n_real == 0 and n_runs == 0
    assert enhanced is False, "drives the If/Else Switch so the enhance branch is skipped"
    assert target_size >= 8, "never 0, which would fail ImageResizeKJv2"
    assert frame_count == clip.shape[0] >= 1
    assert all(not e.get("present") for e in data["entries"])


# ── the fixed target_size mode ───────────────────────────────────────────────

def test_fixed_target_size_ignores_the_face_size():
    imgs, mask = face_track(ZOOM_WIDTHS)
    _clip, data, tgt, *_ = crop_on_width(imgs, mask, 2.0, 10.0, 0.0, 0.3, 0.4,
                                         fixed_target_size=True, target_size=768)
    assert tgt == 768
    assert data["target_size"] == 768 and data["fixed_target_size"] is True


def test_fixed_target_size_is_aligned_up_to_a_multiple_of_32():
    imgs, mask = face_track(ZOOM_WIDTHS)
    _clip, _data, tgt, *_ = crop_on_width(imgs, mask, 2.0, 10.0, 0.0, 0.3, 0.4,
                                          fixed_target_size=True, target_size=700)
    assert tgt == 704 and tgt % 32 == 0


def test_ratio_mode_still_follows_the_crop_window():
    imgs, mask = face_track(ZOOM_WIDTHS)
    _clip, data, tgt, *_ = crop_on_width(imgs, mask, 2.0, 10.0, 0.0, 0.3, 0.4,
                                         fixed_target_size=False, target_size=768)
    assert tgt == max(8, round(data["out_side"] * 2.0))
    assert tgt != 768


# ── the gate's drop breakdown ─────────────────────────────────────────────────

def gate_line(capsys):
    """The '<n>/<N> frames enhanced, <m> dropped: …' line from the console log."""
    lines = [l for l in capsys.readouterr().out.splitlines() if "frames enhanced," in l]
    assert len(lines) == 1, f"expected exactly one gate line, got {lines}"
    return lines[0]


def test_gate_reports_why_each_frame_was_dropped(capsys):
    # 2 tiny, 3 inside the window, 2 huge, 1 with no face at all.
    imgs, mask = face_track([8, 10, 40, 44, 48, 200, 180, None])
    _clip, _data, _tgt, n_real, _runs, _fc, _enh, report = crop_on_width(
        imgs, mask, 2.0, 15.0, 0.0, 1.5, 1.0, min_percent=5.0)

    line = gate_line(capsys)
    assert "3/8 frames enhanced, 5 dropped" in line, line
    assert "1 no face" in line
    assert "2 too large (≥15.00% of frame width)" in line
    assert "2 too small (≤5.00% of frame width)" in line
    assert n_real == 3
    # the same breakdown rides along on the report STRING output
    assert "dropped 5: 1 no face, 2 too large, 2 too small" in report


def test_the_drop_counts_are_exhaustive(capsys):
    imgs, mask = face_track([8, 40, 44, 200, None, 12])
    _clip, _data, _tgt, n_real, *_ = crop_on_width(
        imgs, mask, 2.0, 15.0, 0.0, 1.5, 1.0, min_percent=5.0)
    line = gate_line(capsys)

    import re
    enhanced, dropped = map(int, re.search(r"(\d+)/\d+ frames enhanced, (\d+) dropped", line).groups())
    parts = sum(int(n) for n in re.findall(r"(\d+) (?:no face|too large|too small)", line))
    assert enhanced == n_real
    assert enhanced + dropped == imgs.shape[0]
    assert parts == dropped, "every dropped frame must land in exactly one category"


def test_hysteresis_is_disclosed_in_the_counts(capsys):
    """The thresholds quoted are the ON ones, so with a dead-band they differ from
    the widget values — the line has to say so or the numbers look wrong."""
    imgs, mask = face_track([8, 40, 44, 200])
    crop_on_width(imgs, mask, 2.0, 15.0, 2.0, 1.5, 1.0, min_percent=5.0)
    line = gate_line(capsys)
    assert "≥13.00%" in line and "≤7.00%" in line, line
    assert "hysteresis 2.00%" in line


def test_no_lower_bound_omits_the_too_small_clause(capsys):
    imgs, mask = face_track([8, 40, 200, None])
    crop_on_width(imgs, mask, 2.0, 15.0, 0.0, 1.5, 1.0, min_percent=0.0)
    line = gate_line(capsys)
    assert "too small" not in line, line
    assert "too large" in line


def test_the_no_op_path_still_reports_the_counts(capsys):
    imgs, mask = face_track([200, 200, 200])      # every face over the gate
    _clip, _data, _tgt, n_real, *_ = crop_on_width(
        imgs, mask, 2.0, 15.0, 0.0, 1.5, 1.0, min_percent=5.0)
    assert n_real == 0
    assert "0/3 frames enhanced, 3 dropped" in gate_line(capsys)
