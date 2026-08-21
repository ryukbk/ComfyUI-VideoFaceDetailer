# ComfyUI-VideoFaceDetailer

Selectively upscale and resample **small faces in a video**, then composite them
back into the original footage at their original position and scale — without
touching frames where the face is already large enough.

It is built around ComfyUI's **native SAM 3 / 3.1** tracking nodes and a video
resampler — either the native **LTXV** image-to-video nodes **or the local
**MiniMax H3** joint audio-video model (with **lip-sync to the original audio**;
see [MiniMax H3](#minimax-h3-resampler-with-lip-sync)) — and adds the glue that
none of those provide on their own:

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
     • keeps only frames where the face < max_threshold_percent (of width/height/area)                 │
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
| Native **LTXV** nodes (bundled with ComfyUI) | the video resample pass (LTX workflows) |
| Native **MiniMax H3** nodes (`comfy_extras/nodes_minimax_h3.py`, bundled with a recent ComfyUI) | the video resample pass (H3 workflow): `MiniMaxH3ReferenceToVideo`, `UNETLoader`/`CLIPLoader`/`VAELoader`, `SamplerCustomAdvanced`, etc. |
| [**ComfyUI-KJNodes**](https://github.com/kijai/ComfyUI-KJNodes) | `ImageResizeKJv2`, `GetImageSizeAndCount`, `GetMaskSizeAndCount`, `LazySwitchKJ` |
| [**ComfyUI-VideoHelperSuite**](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | `VHS_LoadVideo`, `VHS_VideoCombine`, `VHS_VideoInfo` |

You also need a SAM 3 / 3.1 checkpoint plus **either** an LTXV checkpoint **or**
the MiniMax H3 models (diffusion model, Qwen3-VL text encoder, and the H3 video +
audio VAEs — [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3))
placed in the usual ComfyUI model folders.

> **The H3 img2img / lip-sync / per-frame-denoise logic is built into this pack**
> as the single `H3FaceRefine` node, so the H3 workflow needs **no third-party node
> packs** beyond core H3 (and KJNodes/VHS as above). Its logic is adapted from the
> community packs credited under
> [Acknowledgements](#acknowledgements).

---

## Example workflows

Drag any of these onto the ComfyUI canvas (they are UI workflow-format JSON):

| File | What it does |
|---|---|
| `workflows/face_enhance_ltx_track_workflow_UI.json` | **Recommended.** Temporally-coherent single LTX pass over one tracked face. |
| `workflows/face_enhance_ltx_track_perrun_workflow_UI.json` | Reference variant that gives **each run its own LTX pass** (no cross-run bridging). Ships configured for up to 2 runs. |
| `workflows/face_enhance_ltx_workflow_UI.json` | Per-frame, per-face variant (uses `SAM3_Detect` union masks; handles multiple faces per frame, no tracking). |
| `workflows/face_enhance_h3_track_workflow_UI.json` | **MiniMax H3 variant** with lip-sync. Same tracked front-end, but the LTX block is replaced by H3 img2img (`MiniMaxH3ReferenceToVideo` → `H3FaceRefine` → `SamplerCustomAdvanced`). See [MiniMax H3](#minimax-h3-resampler-with-lip-sync). |

Set the placeholders before running: the input `video`, the SAM 3 / 3.1
`ckpt_name`, and the LTXV `ckpt_name` (LTX workflows). For the **H3** workflow set
instead: the H3 diffusion model / text encoder / VAEs on the loaders and an
**identity reference image** (`LoadImage`) of the person. Lip-sync uses the source
video's **own audio** automatically — no separate track needed (swap in an
isolated-vocals source on the audio-lock input if you want a cleaner signal).

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
| `threshold_type` | choice | width | which face dimension `max_threshold_percent` measures: **width**, **height**, or **area** |
| `max_threshold_percent` | FLOAT | 10.0 | **upper bound**, as a **percent** (same unit as the `report`) — enhance a frame **only while** the face is *smaller* than this percent of the dimension chosen by `threshold_type` (width/height → that dimension; area → whole-frame area, e.g. `12` = faces under 12% of the frame). Fine grain (`0.01`) so tiny area values like `1.38%` are settable. |
| `min_threshold_percent` | FLOAT | 0.0 | **lower bound**, as a **percent** in the same measure as `threshold_type` (matching `max_threshold_percent`'s units). Faces *smaller* than this are skipped (too tiny to resample usefully). 0 = no lower bound. Enhancement runs only when `min < measure < max`. ⚠️ Setting this **≥** `max_threshold_percent` makes the enable window **empty** (nothing qualifies → the whole video passes through unchanged). |
| `hysteresis_percent` | FLOAT | 0.0 | dead-band around **both** thresholds, as a **percent** (same unit as the thresholds), to stop on/off flicker during a slow zoom. 0 = crisp boundary (default); raise it if a face hovering at the threshold flickers on/off |
| `padding` | FLOAT | 0.3 | context margin around the face box. Keep **low** (0–0.1) if you find LTX enlarges the face (see Limitations) |
| `smooth_alpha` | FLOAT | 1.0 | crop **center** smoothing (EMA). **1.0 (default) = follow the face exactly, no positional lag** (the enhanced face tracks the head). Lower = steadier framing but lags fast head motion (reads as the face being out of sync); drop toward `0.7` only if raw mask noise makes the crop jitter |
| `max_size_deviation` | FLOAT | 0.5 | clamp each frame's crop size to `[median/(1+d), median·(1+d)]`; stops occasional tall/merged masks from engulfing the body |
| `size_smooth_alpha` | FLOAT | 0.4 | crop **size** smoothing (EMA) — the actual *wobble* control, independent of position |
| `resampler` | choice | ltx | which resampler this clip feeds, so it is padded to that model's valid frame-count grid: **ltx** → `8n+1` (LTXVImgToVideo); **minimax_h3** → `17k+5` (MiniMax H3). The padded frame count is what you wire into the resampler's `length` (use the `frame_count` output), so paste-back stays 1:1. Leave `ltx` for the LTX workflows. |

**Outputs:** `face_clip` (IMAGE), `track_data` (FACE_TRACK_DATA), `target_size`
(INT), `enhanced_frames` (INT), `num_runs` (INT), `frame_count` (INT — the
**padded** clip length, i.e. what the resampler's `length` must be; wire it
straight into `MiniMaxH3ReferenceToVideo.length` / `LTXVImgToVideo.length`
instead of a separate `GetImageSizeAndCount` node), `enhanced` (BOOLEAN — True iff
≥1 frame qualified; wire into `LazySwitchKJ.switch` so the enhance branch is
skipped when nothing qualifies. This is the recommended gate — it supersedes
`MaskHasFace`, since "no face" is just one way to get 0 qualifying frames),
`report` (STRING).

**Picking thresholds from your footage:** run the workflow once and read the
`report` output (also printed to the console). It shows, over the frames that
have a face, the **min–max face fraction in all three measures** — e.g.
`faces on 118/124 frames — percent min–max: width 9.1–26.8%, height 12.0–34.5%,
area 1.10–9.20%`. Set `max_threshold_percent` just **above** the largest face
you still want enhanced and `min_threshold_percent` just **below** the smallest,
reading the column for your chosen `threshold_type`. Wire `report` into a
text-preview node (e.g. KJNodes' "Display Any") to keep it on screen.

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
(BOOLEAN, True), `colour_match` (FLOAT, 0.0). **Output:** `images` (IMAGE).

`colour_match` (0 = off, back-compat default) matches the refined face's
per-channel mean/std to the original crop region before compositing, so an
independent resample pass doesn't paste back a subtly brighter/shifted face —
the main lever for a clean **edge/seam match** (recommended `~1.0` for the
generative MiniMax H3 path). See [edges & denoise](#edges-seams-and-denoise).

`blend_mode = mask` (default) composites using the **face-shaped segmentation
alpha** (from the tracked mask), Gaussian-feathered — so only face pixels are
written and the surrounding background is untouched (no rectangular seam, and
any tone/scale drift in the crop margin is not pasted). `rectangle` is the legacy
feathered-square blend. Downsampling uses area interpolation (alias-free);
upscaling uses bicubic. It **errors loudly** if `processed_clip` count ≠ the
crop's frame count, rather than silently pasting faces onto wrong frames.

### Mask Has Face (bool) — `MaskHasFace`
`MASK -> (BOOLEAN has_face, INT frames_with_face)`. True if any frame's mask has
at least `min_pixels` set. **Superseded in the example workflows** by
`FaceTrackCropAndGate.enhanced`, which is a stricter gate: it's True only when
≥1 frame actually falls inside the enable window, whereas `MaskHasFace` fires on
*any* mask pixel (so a non-face SAM3 blob, or a face too large to need
enhancement, would still run the branch). The node is kept for cases where you
specifically want "any face present" rather than "any frame will be enhanced."
It reads the SAM3 mask directly, so place it **outside** the detailer branch.

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

### MiniMax H3 Face Refine — `H3FaceRefine`
One node that turns the local **MiniMax H3** joint audio-video model into a face
resampler for the tracked pipeline. It does three things in the one correct order
(so the workflow stays simple; only the H3 workflow uses it):

1. **img2img inject** — encodes the upscaled face clip into the **video stream** of
   H3's joint AV latent, so the sampler runs genuine img2img (H3's stock nodes start
   from zeros, which would *regenerate* rather than *refine*).
2. **lip-sync** — encodes the **original audio** (the source video's track, or an
   isolated-vocals source) into the audio stream and masks sampling so only video
   denoises while attending to that fixed audio — this shapes the mouth. Relies on
   core H3 honouring the `minimax_h3_lock_audio_clean` transformer option.
3. **per-frame denoise** — varies denoise along time via the latent noise mask,
   strong on the smallest faces (synthesise) and gentle on the largest (preserve).
   The ramp bounds come straight from the gate: the per-frame face measure and the
   `min_threshold_percent` / `max_threshold_percent` window are read from
   `track_data`, so it **follows `threshold_type`** and the exact window you set —
   no separate pixel numbers to tune.

**Inputs:** `model`, `av_latent` (from `MiniMaxH3ReferenceToVideo`), `images` (the
upscaled `face_clip`), `vae` (H3 **video** VAE), `track_data`; and optionally
`audio_vae` (H3 **audio** VAE) + `audio` for lip-sync (omit both to skip it). Plus
the per-frame-denoise widgets: `strength_small_face` (applied at the gate's
`min_threshold_percent`), `strength_large_face` (at `max_threshold_percent`),
`gamma`, `smooth_frames`.
**Outputs:** patched `model` (→ `BasicGuider` / `BasicScheduler`), `av_latent`
(→ `SamplerCustomAdvanced.latent_image`), and a `report` string. Set the overall
strength with `BasicScheduler`'s `denoise`.

### Face Track Audio Slice — `FaceTrackAudioSlice`
Reindexes the source audio to the **gated** face clip so H3 lip-sync matches the
enhanced frames instead of the whole timeline. For each clip frame it takes a
`1/target_fps`-second window from the source at that frame's original time
(`source frame index ÷ source_fps`) and concatenates them in clip order (silence
for pad/absent frames). **Inputs:** `audio`, `track_data`, `source_fps` (wire
`VHS_VideoInfo` `source_fps`), `target_fps` (24 for H3). **Output:** `audio`
aligned 1:1 with the clip — feed it to `H3FaceRefine`'s `audio` and the H3
reference node's `ref_audio`. Mux the **full original** audio at the save node.
For a 24fps source + one contiguous run this is a plain slice; other fps / multiple
runs are resampled per frame onto the 24fps grid.

---

## MiniMax H3 resampler (with lip-sync)

`workflows/face_enhance_h3_track_workflow_UI.json` swaps the LTX block for a local
MiniMax H3 img2img pass that lip-syncs the refined face to the original audio.

**Pipeline** (`workflows/face_enhance_h3_track_workflow_UI.json`):

```
Model chain (each step patches the model; the sampler uses the final one):
  UNETLoader (H3 ref2va) → LoraLoaderModelOnly (turbo) → ModelAttentionBackend ("comfy kitchen attention")
      → H3FaceRefine (model in) → [H3FaceRefine emits the patched model] → BasicGuider + BasicScheduler

Main graph:
  VHS_LoadVideo ─┬─ images ─► SAM3_VideoTrack ─► SAM3_TrackToMask ─ mask ─► FaceTrackCropAndGate
                 │                                                          (resampler = minimax_h3)
                 │                                          ┌──── enhanced (BOOL) ──────────────────┐
                 │                                     face_clip ─┐  │ track_data                    │
                 │                                          ▼      │  │                              │
                 │                                ImageResizeKJv2 (×32)    │                          │
                 │   audio (source) ─► FaceTrackAudioSlice ─(sliced audio)─┐                          │
                 │   ref image ─────► MiniMaxH3ReferenceToVideo ◄──────────┘ │                        │
                 │                        (length ← frame_count, ref_image_size = max)               │
                 │                                positive │ AV latent     │                          │
                 │                                         ▼               │                          │
                 │            H3FaceRefine  (img2img inject + lip-sync + per-frame denoise)           │
                 │                                         │ av_latent                                │
                 │                                         ▼                                           │
                 │            SamplerCustomAdvanced ─► VAEDecode ─ refined faces                      │
                 │                                                     │                               │
                 ├─ original frames ──────► FaceTrackPasteBack (colour_match) ◄────────────────────────┘
                 │                                     │ on_true (LAZY)
                 │      gate.enhanced ─────────► LazySwitchKJ  (on_false = original video)
                 │                                     │
                 └─ audio (source) ───────► VHS_VideoCombine (frame_rate ← source_fps) ─► saved .mp4
```

The switch is driven by `FaceTrackCropAndGate.enhanced` (True iff ≥1 frame
qualified). The gate runs eagerly to produce that boolean, but the whole enhance
branch (Resize → H3 → sampler → decode → paste, plus the H3 model loaders) sits
behind the **lazy** `on_true` — so when no frame qualifies (large-face video,
empty enable window, or no face at all) the branch is **never executed** and the
original video passes through `on_false`. This is why `MaskHasFace` is no longer
needed in the graph: the gate's own decision is a strict superset of "is there a
face" (no face ⇒ 0 frames ⇒ `enhanced=False`).

Also wired (left out above for readability): the H3 **text encoder** → `MiniMaxH3ReferenceToVideo`;
the **video VAE** → `MiniMaxH3ReferenceToVideo` / `H3FaceRefine` / `VAEDecode`; the **audio VAE** →
`MiniMaxH3ReferenceToVideo` and `H3FaceRefine`; and `FaceTrackCropAndGate`'s `track_data` →
`FaceTrackAudioSlice`, `H3FaceRefine` (per-frame denoise) and `FaceTrackPasteBack`.

> **Why `FaceTrackAudioSlice`?** The gate keeps only the small-face frames (often
> several non-contiguous runs) and concatenates them, so the clip fed to H3 is a
> **subset** of the timeline. H3's lip-sync lock aligns audio to video **1:1**, so
> feeding the whole track would sync the generated mouths to the wrong words.
> `FaceTrackAudioSlice` reindexes the source audio to the gated clip's frames
> (per clip frame, a window from the source at that frame's original time), so the
> lip-sync matches. The **full original audio** is still muxed onto the saved video
> at `VHS_VideoCombine`. (This is why this pack needs the slice while Carasibana's
> doesn't — that pack runs *every* frame through H3, so the whole audio aligns.)

**Constraints (handled for you):**
- **Frame count on H3's `17k+5` grid** — set by `resampler = minimax_h3` on the crop
  node; its **`frame_count`** output (the padded length) drives `MiniMaxH3ReferenceToVideo.length`
  **directly** — no `GetImageSizeAndCount` node needed — and pad frames are ignored on
  paste-back.
- **Canvas divisible by 32** — the `ImageResizeKJv2` `divisible_by = 32` widget.
- **24 fps** — H3 interprets `length` at 24 fps, so lip-sync is most accurate on
  24 fps source; the result is muxed with the original audio at the source fps.

**Identity vs. content:** the face clip is the img2img *content* (injected by
`H3FaceRefine`); `ref_image_0` is the *identity* reference (the character image you
generated with). Do **not** feed the clip as `ref_videos` — that regenerates rather
than refines.

> **Set `ref_image_size = max` on `MiniMaxH3ReferenceToVideo`** (the shipped
> value). At `match` the reference is downscaled to the small face canvas and
> barely influences identity — effectively "the ref image is ignored," a common
> gotcha. `max` keeps the reference at up to 2048px (slower, but the identity
> actually takes). Lip-sync is driven by the **source video's own audio**, wired
> into both `ref_audio_0` and the audio lock.

> **The prompt must cite the references, or the model ignores them.** Per
> MiniMax's [reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md),
> a reference only binds when the prompt names it with its tag: `<Picture 1>` =
> `ref_image_0` (the identity image), `<Audio 1>` = `ref_audio_0` (the sliced
> source speech), with `<Subject N>` / `<Video N>` for the other kinds. The
> shipped prompt follows the guide's **structured, six-section** format —
> `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`,
> `overall_soundscape`, `non_diegetic_music` — and cites both references:
>
> ```
> subject_definitions:
> <Subject 1> is the person in <Picture 1>. <Audio 1> is the speech/voice reference for <Subject 1> (S1) …
> …
> retention_analysis:
> <Subject 1> …: fully_preserved - the exact facial features, identity and skin texture from <Picture 1> are retained.
> <Picture 1> …: fully_preserved …
> <Audio 1>: fully_copy - the mouth shapes and speech timing follow this audio exactly for lip-sync.
> …
> ```
>
> `retention_analysis` is the section that tells H3 how tightly to hold each
> reference — `fully_preserved` for the identity/frame, `fully_copy` for the
> locked audio (visible refs also allow `partially_preserved` / `attribute_transfer`
> / `weak_reference`; audio allows `partially_copy` / `reference` / `weak_reference`).
> A generic prompt with no `<Picture N>` / `<Audio N>` tags leaves the identity and
> lip-sync unbound even when the references are connected.

### Packaging the detailer as a subgraph

The H3 workflow ships with a group box titled **"H3 Face Detailer — select this
group, then 'Convert to Subgraph'"** around the whole detailer core (SAM3 →
gate → resize → H3 → decode → paste → `LazySwitchKJ`). To collapse it into a
single reusable subgraph node:

1. Click the group's title bar to select all nodes inside it.
2. Right-click → **Convert to Subgraph** (ComfyUI builds it in your exact
   frontend version's format).
3. The subgraph exposes **8 inputs** and **1 output** — the links that cross the
   group boundary:
   - inputs: `images` (source frames), `model`, `clip`, `vae`, `audio_vae`,
     `reference_image` (identity), `audio` (source), `source_fps`
   - output: `images` (the final composited frames from `LazySwitchKJ`)
4. ComfyUI names the input slots after their source (e.g. `IMAGE`, `VAE`,
   `MODEL`); double-click a slot to rename it to the friendly names above.

`ModelAttentionBackend`, the H3 loaders, `VHS_LoadVideo`, `VHS_VideoInfo`, and
`VHS_VideoCombine` stay **outside** and wire into the subgraph node. (The SAM3
checkpoint stays inside, since SAM3 tracking is part of the detailer.)

> Hand-authoring subgraph JSON directly is intentionally avoided here: ComfyUI's
> subgraph serialization is version-specific, so the reliable, portable way to
> create one is the in-editor **Convert to Subgraph** on the pre-grouped nodes.

### Edges, seams, and denoise

If the pasted face doesn't match the surrounding video at the edges:
- **`colour_match`** on `FaceTrackPasteBack` (≈1.0 for H3) removes tone/brightness
  drift between the independent resample pass and the original — the single most
  effective seam fix.
- **Denoise is an indirect lever:** lower denoise keeps the refined face closer to
  the original in pose/scale/tone, so it lines up under the feather; push it too
  high and the head drifts relative to the body, which no mask can hide. On H3,
  `denoise` lives on `BasicScheduler` (not `SplitSigmas`), and H3's large
  sigma-shift makes small values stronger than they look.
- **`H3FaceRefine`'s per-frame denoise** avoids over-rewriting already-large faces —
  where seams are most visible — by dropping denoise as face size grows.
- **`feather`** / **`padding`** put the seam in hair/background; SAM-style masks
  trace tightly, so lower `feather` if you use them.

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

- **The whole video came through unenhanced + a `WARNING: 0 frames selected for
  enhancement` in the console** → the gate found **no frame inside the enable
  window**, so it emits a **no-op passthrough** (original video preserved) rather
  than crashing. The warning states the cause. Common ones: no face was small
  enough (raise `max_threshold_percent`), `min_threshold_percent ≥ max_threshold_percent`
  so the window is empty (lower `min_threshold_percent`, usually back to `0`),
  `hysteresis_percent` too wide, or the SAM3 track was empty (check `object_indices`).
  Note: because the gate runs eagerly upstream, a downstream `LazySwitchKJ`
  cannot skip it — this graceful no-op is what makes the skip-when-no-face case
  work end to end. `threshold_type` only selects which dimension the single
  `max_threshold_percent` measures (width/height/area).
- **`ValueError: height and width must be > 0` at the Resize (ImageResizeKJv2)
  node** → this happened in older graphs when 0 frames qualified: the gate emitted
  a tiny no-op dummy that Resize's `divisible_by` (e.g. 32) rounded down to 0
  (`16 - 16%32 = 0`). **Fixed** by driving `LazySwitchKJ.switch` from
  `FaceTrackCropAndGate.enhanced` (slot 6): when no frame qualifies the branch —
  including Resize — is skipped entirely, so the dummy never reaches Resize.
  Re-import the current workflow (switch ← `enhanced`, **not** `MaskHasFace`) and
  do a **full ComfyUI restart**.
- **The enhance branch runs even though "there's no face"** → in older graphs the
  switch was driven by `MaskHasFace`, which counts *any* mask pixel, so a non-face
  SAM3 blob made it run. Now the switch is driven by the gate's `enhanced` boolean,
  which is True only when ≥1 frame actually falls inside the enable window — so a
  non-qualifying clip (too-large faces, empty window, no face) skips the branch.
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

Built on top of ComfyUI's native **SAM 3 / 3.1**, **LTXV**, and **MiniMax H3**
nodes, and designed to interoperate with
[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) and
[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite).

The single `H3FaceRefine` node folds together — and adapts to this pack's
tracked-face pipeline — techniques worked out by the community:
- **video-latent img2img injection** and **per-frame denoise via the latent noise
  mask**, from
  [**ComfyUI-H3-FaceRefine**](https://github.com/Carasibana/ComfyUI-H3-FaceRefine)
  by Carasibana (MIT) — originally its separate `H3InjectVideoLatent` and
  `H3PerFrameDenoise` nodes. Here they are merged into one node, and the per-frame
  denoise ramp is driven by this pack's gate window (`min_threshold_percent` /
  `max_threshold_percent`, following `threshold_type`) rather than the original's
  absolute face-pixel sizes.
- the **native-audio lock** for lip-sync (the `minimax_h3_lock_audio_clean`
  transformer option), from
  [**ComfyUI-H3-NativeAudioLock**](https://github.com/Shrek3OnVH5/MiniMax-H3-NativeAudio-MusicVideo-Workflow)
  by Shrek3OnVH5.

Those upstream packs are MIT-licensed; the adapted code is redistributed here
under this repository's AGPL-3.0, with attribution.
