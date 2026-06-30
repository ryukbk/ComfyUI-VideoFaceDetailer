# ComfyUI-VideoFaceDetailer

Selectively upscale and resample **small faces in a video**, then composite them
back into the original footage at their original position and scale — without
touching frames where the face is already large enough.

It is built around ComfyUI's **native SAM 3 / 3.1** tracking nodes and a video
resampler (the examples use the native **LTXV** image-to-video nodes), and adds
the glue that none of those provide on their own:

- a **size gate** — only enhance a face while it is smaller than a chosen
  fraction of the frame width;
- **temporally-coherent, jitter-free cropping** of a tracked face across frames;
- **paste-back** that returns each enhanced face to its exact original location
  and scale.

> **Status:** functional and unit-tested, but not yet widely battle-tested in
> the wild. Please read the **Limitations** section and report issues.

---

## Why this exists

The "detail a small face, then put it back" pattern (FaceDetailer-style) is easy
for stills but awkward for video, because:

1. **Native ComfyUI has no conditional/size branching** — nothing decides
   "enhance this frame only if the face is below N% of the width".
2. **Crop nodes don't follow a tracked face across frames** without scale/position
   jitter, which makes the enhanced result visibly *wobble*.
3. **Video resamplers need fixed-size, fixed-length clips**, and pasting the
   result back at the correct per-frame location/scale is fiddly.

This pack handles all three.

---

## How it works

```
                                        ┌──────────────── preview (optional) ───────────────┐
VHS_LoadVideo ── images ──┐             │                                                    │
                          ▼             ▼                                                    │
   SAM3_VideoTrack → SAM3_TrackToMask → (mask track of ONE face)                             │
                          │                                                                  │
                          ▼                                                                  │
   Face Track Crop & Gate (coherent)                                                         │
     • keeps only frames where the face < max_width_fraction of frame width                  │
     • crops each kept frame, smoothed in position AND size (no wobble)                      │
     • outputs face_clip + target_size (= crop size × upscale_ratio) + num_runs              │
                          │                                                                  │
        face_clip ────────┼──────────────► Resize (to target_size)  ── UPSCALE              │
        target_size ──────┘                        │                                         │
                                                    ▼                                         │
                                          LTXV img2video + KSampler  ── RESAMPLE              │
                                                    │                                         │
                                                    ▼                                         │
                                                VAEDecode                                     │
                                                    │                                         │
   Face Track Paste Back (coherent) ◄───────────────┘                                         │
     • downsamples each processed face back to its original crop size  ── DOWNSAMPLE         │
     • composites it onto the original frame at the original position (feathered)            │
                          │                                                                  │
                          ▼                                                                  │
                   VHS_VideoCombine ──────────────────────────────────────────────────────►┘
```

The size gate runs **per frame with hysteresis**, so a face that crosses the
threshold (e.g. the camera zooms in, the face grows past the limit, then shrinks
again) is enhanced **only while it is small** — large-face frames pass through
untouched. Those enhanced frames may form several disjoint "runs"; see
[Multiple runs](#multiple-runs).

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<you>/ComfyUI-VideoFaceDetailer
# then FULLY restart ComfyUI (stop the process and relaunch — NOT the
# "Restart" button in ComfyUI Manager, which may keep stale modules loaded)
```

The custom nodes themselves depend only on `torch` and `numpy` (already present
in ComfyUI). The **example workflows** additionally require:

| Dependency | Used for |
|---|---|
| Native **SAM 3 / 3.1** nodes (bundled with recent ComfyUI) | `SAM3_VideoTrack`, `SAM3_TrackToMask`, `SAM3_TrackPreview` |
| Native **LTXV** nodes (bundled with ComfyUI) | the video resample pass |
| [**ComfyUI-KJNodes**](https://github.com/kijai/ComfyUI-KJNodes) | `ImageResizeKJv2`, `GetImageSizeAndCount`, `GetMaskSizeAndCount` |
| [**ComfyUI-VideoHelperSuite**](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | `VHS_LoadVideo`, `VHS_VideoCombine`, `VHS_VideoInfo` |

You also need a SAM 3 / 3.1 checkpoint and an LTXV checkpoint placed in the usual
ComfyUI model folders.

---

## Example workflows

Drag any of these onto the ComfyUI canvas (they are UI workflow-format JSON):

| File | What it does |
|---|---|
| `workflows/face_enhance_ltx_track_workflow_UI.json` | **Recommended.** Temporally-coherent single LTX pass over one tracked face. |
| `workflows/face_enhance_ltx_track_perrun_workflow_UI.json` | Reference variant that gives **each run its own LTX pass** (no cross-run bridging). Ships configured for up to 2 runs. |
| `workflows/face_enhance_ltx_workflow_UI.json` | Per-frame, per-face variant (uses `SAM3_Detect` union masks; handles multiple faces per frame, no tracking). |

Set the placeholders before running: the input `video`, the SAM 3 / 3.1
`ckpt_name`, and the LTXV `ckpt_name`.

---

## Nodes

### Face Track Crop & Gate (coherent) — `FaceTrackCropAndGate`
Crops one tracked face across the video, gated by size, ready for upscaling.

**Inputs**

| name | type | default | meaning |
|---|---|---|---|
| `images` | IMAGE | — | the video frames |
| `mask_track` | MASK | — | per-frame mask of **one** tracked face (from `SAM3_TrackToMask`) |
| `upscale_ratio` | FLOAT | 2.0 | crop is upscaled by this for resampling; `target_size = crop_size × ratio`. Paste-back undoes it exactly. |
| `threshold_type` | choice | width | which face dimension the gate uses: **width**, **height**, or **area**. Selects which of the three parameters below is active (the UI shows only that one). |
| `max_width_fraction` | FLOAT | 0.10 | *(threshold_type=width)* enhance a frame **only while** the face is narrower than this fraction of the frame **width** |
| `max_height_fraction` | FLOAT | 0.10 | *(threshold_type=height)* enhance **only while** the face is shorter than this fraction of the frame **height** |
| `max_area_percent` | FLOAT | 10.0 | *(threshold_type=area)* enhance **only while** the face bbox occupies less than this **percent of the whole frame area** (e.g. 12.1 = faces smaller than 12.1% of the frame) |
| `min_threshold_percent` | FLOAT | 0.0 | **lower bound**, as a percent in the same measure as `threshold_type`. Faces *smaller* than this are skipped (too tiny to resample usefully). 0 = no lower bound. Enhancement runs only when `min < measure < max`. |
| `hysteresis` | FLOAT | 0.02 | dead-band around **both** thresholds (same normalized units as the chosen measure) to stop on/off flicker during a slow zoom |
| `padding` | FLOAT | 0.3 | context margin around the face box. Keep **low** (0–0.1) if you find LTX enlarges the face (see Limitations) |
| `smooth_alpha` | FLOAT | 0.4 | crop **center** smoothing (EMA). **1.0 = follow the face exactly, no positional lag** |
| `max_size_deviation` | FLOAT | 0.5 | clamp each frame's crop size to `[median/(1+d), median·(1+d)]`; stops occasional tall/merged masks from engulfing the body |
| `size_smooth_alpha` | FLOAT | 0.4 | crop **size** smoothing (EMA) — the actual *wobble* control, independent of position |

**Outputs:** `face_clip` (IMAGE), `track_data` (FACE_TRACK_DATA), `target_size`
(INT), `enhanced_frames` (INT), `num_runs` (INT).

> **Anti-wobble tip:** for a steady face that the mask jitters on, use
> `smooth_alpha = 1.0` (exact position) and lower `size_smooth_alpha`
> (≈0.3) with low `padding`. Increasing `padding` also reduces apparent wobble
> but lets the resampler reframe the face — prefer the smoothing controls.

### Face Track Paste Back (coherent) — `FaceTrackPasteBack`
Downsamples each processed face back to its original crop size and composites it
onto the original frame at the original location.

**Inputs:** `original_images` (IMAGE), `processed_clip` (IMAGE, the resampled
faces, **same count/order** as the crop output), `track_data` (FACE_TRACK_DATA),
`feather` (FLOAT, 0.15), `blend_mode` (choice, **mask**), `only_present_frames`
(BOOLEAN, True). **Output:** `images` (IMAGE).

`blend_mode = mask` (default) composites using the **face-shaped segmentation
alpha** (from the tracked mask), Gaussian-feathered — so only face pixels are
written and the surrounding background is untouched (no rectangular seam, and
any tone/scale drift in the crop margin is not pasted). `rectangle` is the legacy
feathered-square blend. Downsampling uses area interpolation (alias-free);
upscaling uses bicubic. It **errors loudly** if `processed_clip` count ≠ the
crop's frame count, rather than silently pasting faces onto wrong frames.

### Face Track Select Run (per-run) — `FaceTrackSelectRun`
Extracts one contiguous run from a multi-run clip for an independent LTX pass.
Out-of-range `run_index` yields a harmless 1-frame no-op, so you can safely wire
more branches than a clip has runs. See [Multiple runs](#multiple-runs).

### Face Track Run Count — `FaceTrackRunCount`
Reports `num_runs` and `enhanced_frames` from `track_data`. (The same `num_runs`
is also an output of the crop node.) Run once to learn how many LTX passes the
per-run workflow needs.

### Per-frame variants — `FaceCropAndGate` / `FacePasteBack`
Operate on `SAM3_Detect` per-frame union masks without tracking. They split each
frame's mask into connected components and enhance **every** face below the size
threshold (multiple faces per frame). Less temporally coherent than the track
nodes; use when you don't have/ want a single tracked subject.

### Face Size Gate (Mask) — `FaceSizeGateMask`
A simple primitive: zero out per-frame masks whose face bbox is ≥
`max_width_fraction` of the frame width. Pairs with KJNodes'
`FilterZeroMasksAndCorrespondingImages` for custom one-crop-per-frame pipelines.

---

## Multiple runs

If the face crosses the size threshold more than once (small → large → small),
the enhanced frames split into several **runs**. The default workflow concatenates
them into **one** LTX pass; the crop node prints how many runs it found and
exposes `num_runs`. A single pass is fine for most footage but will *bridge* the
gap between runs (the resampler interpolates across the discontinuity).

For strict per-run coherence, use
`face_enhance_ltx_track_perrun_workflow_UI.json`: it sends each run through its
own LTX pass via `FaceTrackSelectRun(run_index = 0, 1, …)` and chains the
paste-backs. Read `num_runs` (run once), then provision that many branches.
Extra branches are safe no-ops.

> ComfyUI graphs are static, so the number of LTX passes cannot resize itself to
> the run count automatically — you provision a fixed number of branches.

---

## Tips & troubleshooting

- **"stack expects a non-empty TensorList"** at the Resize node → the gate
  selected **0 frames** (no face was small enough, or `mask_track` was empty).
  The node now raises a clear message instead; raise `max_width_fraction`, lower
  `hysteresis`, or check the SAM3 track / `object_indices`.
- **A node shows up red / "UNKNOWN" after updating** → do a **full ComfyUI
  restart** (not the Manager restart) and clear `__pycache__`.
- **The resampled face looks bigger than the original** → high `padding` gives
  the video model room to reframe/enlarge the face. Lower `padding`, lower the
  LTX denoise, or soften the prompt.
- **`upscale_ratio` seems ignored** → make sure you're editing it on the crop
  node itself (it's a widget), not a stale value upstream.

---

## Limitations

- **Not a turnkey, heavily field-tested pack yet.** Logic is covered by unit
  tests (size gating, run splitting, crop/paste round-trip, count guards,
  interpolation); end-to-end renders depend on your models and ComfyUI build.
- **One tracked face per branch.** Track multiple faces by duplicating the
  crop→resample→paste branch with a different `SAM3_TrackToMask` object index and
  chaining the paste-backs.
- **The resampler can change the face** — it's a generative pass. Use low denoise
  and low padding to keep identity/scale stable.
- **LTX clip-length rules:** the crop node pads its clip to a valid LTX length
  (`8n+1`) automatically; keep the example's `GetImageSizeAndCount → length`
  wiring intact so paste-back counts line up.
- **Widget order matters** when hand-editing workflow JSON — values are
  positional.

---

## License

GNU Affero General Public License v3.0 (AGPL-3.0) — see `LICENSE`.

## Acknowledgements

Built on top of ComfyUI's native **SAM 3 / 3.1** and **LTXV** nodes, and designed
to interoperate with [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
and [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite).
