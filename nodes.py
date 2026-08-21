"""
Face size-gated, per-face crop / paste-back custom nodes for ComfyUI.

Why these exist (verified against current ComfyUI + KJNodes + SAM3 source):
  * Nothing native gates a face by "< X% of screen width".
  * SAM3_Detect(individual_masks=True) concatenates every object across every
    frame into one flat [sum_objects, H, W] batch with NO frame mapping, so the
    KJNodes BatchCropFromMask/BatchUncrop pair cannot do true per-face crop and
    paste-back-into-the-right-frame for multi-face video.

So this module provides:

  FaceSizeGateMask   - simple per-frame gate on a [B,H,W] mask batch. Keeps a
                       frame's (union) mask only if its bbox width < fraction of
                       the frame width. Pairs with KJNodes
                       FilterZeroMasksAndCorrespondingImages for a one-crop-per-
                       frame workflow.

  FaceCropAndGate    - TRUE per-face. Takes video frames + per-frame union masks
                       ([B,H,W] from SAM3_Detect with individual_masks=False),
                       splits each frame's mask into connected components (one per
                       face), keeps only faces narrower than max_width_fraction of
                       the frame width, and outputs a batch of square-padded face
                       crops plus FACE_CROP_DATA describing where each came from.

  FacePasteBack      - takes the (externally upscaled + LTX-resampled) face crops
                       in the SAME order, plus the original frames and the
                       FACE_CROP_DATA, and composites each face back into its
                       source frame at the original location and scale, with a
                       feathered edge. Handles multiple faces per frame.

Install: clone this repo into ComfyUI/custom_nodes/ComfyUI-VideoFaceDetailer/
and fully restart ComfyUI.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resize_hwc(img_hwc, out_h, out_w):
    """Resize an [H,W,C] float image, choosing the interpolation by direction.

    Downsampling uses 'area' (averages source pixels — alias-free, the correct
    choice when shrinking; bilinear undersamples and causes frame-to-frame
    detail shimmer). Upsampling uses 'bicubic' (sharper than bilinear). Returns
    [out_h, out_w, C].
    """
    in_h, in_w = img_hwc.shape[0], img_hwc.shape[1]
    x = img_hwc.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
    shrinking = (out_h * out_w) < (in_h * in_w)
    if shrinking:
        y = F.interpolate(x, size=(out_h, out_w), mode="area")
    else:
        y = F.interpolate(x, size=(out_h, out_w), mode="bicubic", align_corners=False)
    return y[0].permute(1, 2, 0)


def _gaussian_blur_2d(mask_hw, radius):
    """Separable Gaussian blur of a 2D [H,W] float mask. radius<=0 -> unchanged.
    Used to soften a face-shaped alpha so the composite has no hard mask edge."""
    r = int(radius)
    if r <= 0:
        return mask_hw
    sigma = max(0.5, r / 2.0)
    xs = torch.arange(-r, r + 1, dtype=torch.float32, device=mask_hw.device)
    k = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    k = k / k.sum()
    x = mask_hw.unsqueeze(0).unsqueeze(0)                     # [1,1,H,W]
    kh = k.view(1, 1, 1, -1)
    kv = k.view(1, 1, -1, 1)
    x = F.conv2d(x, kh, padding=(0, r))
    x = F.conv2d(x, kv, padding=(r, 0))
    return x[0, 0].clamp_(0.0, 1.0)


def _connected_components(mask_bool):
    """Label connected components (4-connectivity) of a 2D boolean numpy array.

    Dependency-free (no scipy) iterative flood fill. Returns a list of
    (y0, x0, y1, x1) inclusive bounding boxes, one per component. Suitable for
    the handful of faces present in a frame.
    """
    return [b[:4] for b in _connected_components_with_area(mask_bool)]


def _connected_components_with_area(mask_bool):
    """Like _connected_components but each entry is (y0, x0, y1, x1, area)."""
    H, W = mask_bool.shape
    visited = np.zeros((H, W), dtype=bool)
    boxes = []
    ys, xs = np.nonzero(mask_bool)
    pts = list(zip(ys.tolist(), xs.tolist()))
    pset = set(pts)
    for (sy, sx) in pts:
        if visited[sy, sx]:
            continue
        # BFS flood fill
        stack = [(sy, sx)]
        visited[sy, sx] = True
        miny = maxy = sy
        minx = maxx = sx
        area = 0
        while stack:
            y, x = stack.pop()
            area += 1
            if y < miny: miny = y
            if y > maxy: maxy = y
            if x < minx: minx = x
            if x > maxx: maxx = x
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] \
                        and (ny, nx) in pset:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        boxes.append((miny, minx, maxy, maxx, area))
    return boxes


def _measure(bw, bh, area, kind):
    if kind == "bbox_width":
        return bw
    if kind == "bbox_diagonal":
        return (bw * bw + bh * bh) ** 0.5
    return float(area) ** 0.5  # area_sqrt


# ─────────────────────────────────────────────────────────────────────────────
# Resampler clip-length grids
# ─────────────────────────────────────────────────────────────────────────────
# The gated face clip is padded up to a length the chosen resampler accepts and
# returns unchanged, so FaceTrackPasteBack can keep processed frames 1:1 with the
# crop entries. Two resamplers are supported:
#   * "ltx"        — LTXVImgToVideo decodes (T-1)*8+1 frames, i.e. valid lengths
#                    are 8n+1: 1, 9, 17, 25, …
#   * "minimax_h3" — MiniMax H3 (EmptyMiniMaxH3LatentAV / MiniMaxH3ReferenceToVideo)
#                    packs 17 pixel frames per latent frame on a 17k+5 grid:
#                    5, 22, 39, 56, 73, 90, 107, 124, … (min length 5).
_RESAMPLERS = ("minimax_h3", "ltx")


def _next_valid_clip_len(n, resampler="ltx"):
    """Smallest valid clip length >= n for the resampler's temporal grid.

    LTX requires 8n+1; MiniMax H3 requires 17k+5 (minimum 5). Used to pad the
    gated face clip so the resampler returns exactly that many frames.
    """
    n = int(n)
    if resampler == "minimax_h3":
        if n <= 5:
            return 5
        k = (n - 5 + 16) // 17  # ceil((n-5)/17)
        return 17 * k + 5
    # "ltx" (default)
    if n <= 1:
        return 1
    return ((n - 1 + 7) // 8) * 8 + 1


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: simple per-frame gate (union mask)
# ─────────────────────────────────────────────────────────────────────────────

class FaceSizeGateMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "max_width_fraction": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.005}),
                "measure": (["bbox_width", "bbox_diagonal", "area_sqrt"], {"default": "bbox_width"}),
            },
        }

    RETURN_TYPES = ("MASK", "INT", "INT")
    RETURN_NAMES = ("gated_masks", "kept_count", "frame_width")
    FUNCTION = "gate"
    CATEGORY = "masking/face_gate"
    DESCRIPTION = ("Zero out per-frame masks whose face bbox is >= "
                   "max_width_fraction of the frame width.")

    def gate(self, masks, max_width_fraction, measure="bbox_width"):
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        B, H, W = masks.shape
        threshold = max_width_fraction * W
        out = masks.clone()
        kept = 0
        for i in range(B):
            m = masks[i] > 0.5
            if not bool(m.any()):
                out[i].zero_()
                continue
            rows = torch.any(m, dim=1)
            cols = torch.any(m, dim=0)
            ys = torch.nonzero(rows, as_tuple=False)
            xs = torch.nonzero(cols, as_tuple=False)
            bw = int(xs[-1] - xs[0] + 1)
            bh = int(ys[-1] - ys[0] + 1)
            size = _measure(bw, bh, int(m.sum()), measure)
            if size < threshold:
                kept += 1
            else:
                out[i].zero_()
        return (out, kept, W)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: true per-face crop + gate
# ─────────────────────────────────────────────────────────────────────────────

class FaceCropAndGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # [B,H,W,C] video frames
                "masks": ("MASK",),    # [B,H,W] per-frame UNION face masks (SAM3 individual_masks=False)
                "max_width_fraction": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.005}),
                "measure": (["bbox_width", "bbox_diagonal", "area_sqrt"], {"default": "bbox_width"}),
                "padding": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05,
                                      "tooltip": "Expand each face box by this fraction before cropping (context for the model)."}),
                "crop_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8,
                                      "tooltip": "Square size each face crop is resized to before enhancement."}),
                "min_face_px": ("INT", {"default": 8, "min": 1, "max": 4096, "step": 1,
                                        "tooltip": "Ignore components smaller than this (noise)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FACE_CROP_DATA", "INT")
    RETURN_NAMES = ("face_crops", "face_data", "face_count")
    FUNCTION = "crop"
    CATEGORY = "masking/face_gate"
    DESCRIPTION = ("Split per-frame union masks into individual faces, keep only "
                   "those narrower than max_width_fraction of frame width, and "
                   "output a batch of square crops + placement data.")

    def crop(self, images, masks, max_width_fraction, measure, padding, crop_size, min_face_px):
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        B, H, W, C = images.shape
        threshold = max_width_fraction * W
        crops = []
        entries = []  # dicts: frame, x0,y0,x1,y1 (padded crop box in source px)
        for i in range(min(B, masks.shape[0])):
            m = (masks[i] > 0.5).cpu().numpy()
            if not m.any():
                continue
            for (y0, x0, y1, x1) in _connected_components(m):
                bw = x1 - x0 + 1
                bh = y1 - y0 + 1
                if bw < min_face_px and bh < min_face_px:
                    continue
                area = int(m[y0:y1 + 1, x0:x1 + 1].sum())
                if _measure(bw, bh, area, measure) >= threshold:
                    continue  # face already big enough -> skip
                # pad box, clamp to frame
                px = int(bw * padding)
                py = int(bh * padding)
                cx0 = max(0, x0 - px)
                cy0 = max(0, y0 - py)
                cx1 = min(W, x1 + 1 + px)
                cy1 = min(H, y1 + 1 + py)
                crop = images[i, cy0:cy1, cx0:cx1, :]  # [h,w,C]
                crop_r = F.interpolate(crop.permute(2, 0, 1).unsqueeze(0),
                                       size=(crop_size, crop_size),
                                       mode="bilinear", align_corners=False)
                crops.append(crop_r[0].permute(1, 2, 0))
                entries.append({"frame": i, "x0": cx0, "y0": cy0, "x1": cx1, "y1": cy1})

        if len(crops) == 0:
            raise ValueError(
                "FaceCropAndGate: 0 faces selected for enhancement. Either no face "
                "was detected (check SAM3_Detect / the 'face' prompt), or every "
                "detected face is at/above max_width_fraction "
                f"({max_width_fraction * 100:.1f}% = {max_width_fraction * W:.0f}px "
                f"of the {W}px-wide frame) and was skipped as 'already big enough'. "
                "Emitting an empty image batch would crash the downstream Resize/LTX "
                "nodes (torch.stack on empty), so this is raised instead. Raise "
                "max_width_fraction if you want larger faces enhanced."
            )

        face_crops = torch.stack(crops, dim=0)
        face_data = {"entries": entries, "orig_shape": (B, H, W, C)}
        return (face_crops, face_data, len(entries))


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: paste processed faces back
# ─────────────────────────────────────────────────────────────────────────────

class FacePasteBack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),     # [B,H,W,C]
                "processed_faces": ("IMAGE",),      # [N, h, w, C] in SAME order as face_data entries
                "face_data": ("FACE_CROP_DATA",),
                "feather": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01,
                                      "tooltip": "Feather width as fraction of crop size for seamless blending."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "paste"
    CATEGORY = "masking/face_gate"
    DESCRIPTION = ("Composite each processed face back into its source frame at "
                   "the original location and scale (handles multiple faces per "
                   "frame). Processed faces must be in face_data order.")

    def paste(self, original_images, processed_faces, face_data, feather):
        out = original_images.clone()
        entries = face_data.get("entries", [])
        N = processed_faces.shape[0]
        if N != len(entries):
            # Be loud but non-fatal: only paste the overlap.
            print(f"[FacePasteBack] WARNING: {N} processed faces but "
                  f"{len(entries)} entries; pasting min().")
        for idx in range(min(N, len(entries))):
            e = entries[idx]
            f = e["frame"]
            x0, y0, x1, y1 = e["x0"], e["y0"], e["x1"], e["y1"]
            tw, th = x1 - x0, y1 - y0
            if tw <= 0 or th <= 0:
                continue
            face = processed_faces[idx].permute(2, 0, 1).unsqueeze(0)  # [1,C,h,w]
            face_r = F.interpolate(face, size=(th, tw), mode="bilinear", align_corners=False)[0]
            face_r = face_r.permute(1, 2, 0).to(out.dtype).to(out.device)  # [th,tw,C]

            # Feathered alpha (1 in center, ramps to 0 at edges)
            fpx = max(1, int(min(th, tw) * feather))
            alpha = torch.ones(th, tw, dtype=out.dtype, device=out.device)
            if fpx > 0:
                ramp = torch.linspace(0, 1, steps=fpx, dtype=out.dtype, device=out.device)
                alpha[:fpx, :] *= ramp.view(-1, 1)
                alpha[-fpx:, :] *= ramp.flip(0).view(-1, 1)
                alpha[:, :fpx] *= ramp.view(1, -1)
                alpha[:, -fpx:] *= ramp.flip(0).view(1, -1)
            alpha = alpha.unsqueeze(-1)  # [th,tw,1]

            region = out[f, y0:y1, x0:x1, :]
            out[f, y0:y1, x0:x1, :] = region * (1 - alpha) + face_r * alpha
        return (out,)


# ─────────────────────────────────────────────────────────────────────────────
# Temporally-coherent track variant
# ─────────────────────────────────────────────────────────────────────────────
#
# Pair with SAM3: SAM3_VideoTrack -> SAM3_TrackToMask (select ONE tracked face
# via object_indices, e.g. "0") -> [N,H,W] per-frame mask track for that face.
#
# FaceTrackCropAndGate builds a temporally-stable clip for that tracked face,
# gating PER FRAME by face size:
#   - A frame is enhanced ONLY while the face is below max_width_fraction of the
#     frame width. If the camera zooms in and the face grows past the threshold,
#     those frames are dropped from the clip and left untouched in the output
#     (this is the whole point: don't resample a face that's already big). When
#     the face shrinks again, enhancement resumes.
#   - Hysteresis (a dead-band around the threshold) prevents flicker when the
#     face hovers right at the boundary during a slow zoom: once enhancement
#     turns off it won't turn back on until the face drops clearly below
#     threshold, and vice-versa.
#   - The crop window SIZE is computed from the ENHANCED (small-face) frames
#     only, so a large zoomed-in face never bloats the window and steals
#     resolution from the small-face frames.
#   - Per-frame window CENTER follows the face, exponentially smoothed to kill
#     crop jitter.
# It emits target_size = window * ratio so an external upscaler (KJNodes Resize
# Image v2) enlarges by exactly the float ratio. FaceTrackPasteBack then resizes
# the processed clip back to the native window — undoing the ratio precisely —
# and composites only the enhanced frames.
#
# NOTE: tracking itself never stops when the face grows — SAM3 follows the face
# by appearance regardless of size. It is purely this node's per-frame gate that
# decides which frames to resample.
#
# Coherence comes from (a) the stable smoothed window and (b) running the
# enhanced frames through one LTX vid2vid pass. If the face crosses the
# threshold multiple times, the enhanced frames form multiple runs concatenated
# into one clip; for a single zoom-in (the common case) they are one contiguous
# run, so there is no internal seam.
# One tracked face per branch; for multiple faces, duplicate the branch with a
# different SAM3_TrackToMask object index.

def _frame_face_box(mask_bool):
    """Bbox (x0,y0,x1,y1 inclusive) + centroid of the FACE in a single-frame mask.

    Uses the LARGEST connected component, not the global pixel min/max. SAM3
    masks occasionally include stray pixels or a faint secondary blob (neck/body/
    a second detection) on some frames; a global bounding box would then stretch
    to span the face AND the stray region, producing a crop that suddenly engulfs
    the body on those frames. Taking the largest component isolates the face.
    Returns (x0, y0, x1, y1, cx, cy) over the face component, or None if empty.
    """
    if not mask_bool.any():
        return None
    comps = _connected_components_with_area(mask_bool)  # (y0,x0,y1,x1,area)
    if not comps:
        return None
    # Largest component by pixel area = the face.
    y0, x0, y1, x1, _area = max(comps, key=lambda c: c[4])
    # Centroid of THAT component's bounding box (sufficient and stable for the
    # crop center; avoids re-scanning pixels).
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return (int(x0), int(y0), int(x1), int(y1), float(cx), float(cy))


class FaceTrackCropAndGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask_track": ("MASK",),  # [N,H,W] per-frame mask of ONE tracked face
                "upscale_ratio": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1,
                                            "tooltip": "Face is upscaled by this ratio for resampling, then "
                                                       "restored to original size on paste-back. Wire a "
                                                       "FloatConstant here to control it."}),
                "threshold_type": (["width", "height", "area"], {"default": "width",
                                    "tooltip": "Which face dimension the percentages below measure: 'width' -> "
                                               "percent of frame width; 'height' -> percent of frame "
                                               "height; 'area' -> percent of the whole frame area."}),
                "max_threshold_percent": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.01,
                                           "tooltip": "Upper bound, as a PERCENT (same unit as the size report): "
                                                      "enhance a frame ONLY while the face is SMALLER than this "
                                                      "percent of the dimension chosen by threshold_type "
                                                      "(width/height -> that dimension; area -> whole-frame area). "
                                                      "E.g. 12 with threshold_type=area = faces under 12% of the "
                                                      "frame. Read the node's `report` and set this just above the "
                                                      "largest face you want enhanced."}),
                "min_threshold_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01,
                                             "tooltip": "Lower bound, as a PERCENT in the same measure as "
                                                        "threshold_type (matching max_threshold_percent's units). "
                                                        "Faces SMALLER than this are skipped (too tiny to resample "
                                                        "usefully). 0 = no lower bound. Enhancement runs only when "
                                                        "min < measure < max. Set just below the smallest face in "
                                                        "the `report`."}),
                "hysteresis_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.01,
                                         "tooltip": "Dead-band around the thresholds, as a PERCENT (same unit as "
                                                    "max/min_threshold_percent). Stops on/off flicker when the "
                                                    "face hovers at the boundary. ON below (max - hysteresis), "
                                                    "OFF at/above (max + hysteresis)."}),
                "padding": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.05,
                                      "tooltip": "Context margin around the face box. LOW (0.1 default) puts "
                                                 "more of the crop on the face -> more pixels/detail at the same "
                                                 "target_size (crisper). Raise it if the face gets clipped on fast "
                                                 "motion or the model needs more context; very high can let the "
                                                 "resampler reframe/enlarge the face."}),
                "smooth_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                           "tooltip": "Crop CENTER smoothing (EMA). 1.0 (default) = follow the face "
                                                      "exactly, no positional lag (recommended — the enhanced face "
                                                      "tracks the head). Lower = steadier framing but lags fast head "
                                                      "moves, which reads as the face being out of sync. Drop toward "
                                                      "0.7 only if raw mask noise makes the crop jitter."}),
                "max_size_deviation": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.05,
                                        "tooltip": "How far a single frame's crop size may deviate from the "
                                                   "clip's median face size, as a fraction. With 0.5, each "
                                                   "frame's crop side is clamped to [median/1.5, median*1.5]. "
                                                   "Prevents occasional tall/merged masks (face+neck/body) "
                                                   "from producing crops that engulf the body. Lower = stricter "
                                                   "(more uniform, risk of cropping a genuinely-grown face); "
                                                   "higher = looser; very high effectively disables the clamp."}),
                "size_smooth_alpha": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Crop SIZE smoothing (EMA), separate from center. This is the "
                                                   "actual wobble fix: it damps the per-frame mask-bbox jitter "
                                                   "that makes the face pulse in scale. Lower = steadier size "
                                                   "(less wobble) but slower to follow genuine size changes; 1.0 "
                                                   "= raw per-frame size (max wobble). Tune this for wobble, "
                                                   "leave smooth_alpha=1.0 to keep position exact."}),
                "resampler": (list(_RESAMPLERS), {"default": "minimax_h3",
                                        "tooltip": "Which resampler this clip feeds, so it is padded to that "
                                                   "model's valid frame-count grid: 'minimax_h3' (default) -> "
                                                   "17k+5 (MiniMax H3 EmptyMiniMaxH3LatentAV / "
                                                   "MiniMaxH3ReferenceToVideo); 'ltx' -> 8n+1 (LTXVImgToVideo). "
                                                   "Set 'ltx' when feeding the LTX workflows. The padded frame "
                                                   "count (frame_count output) is what you wire into the "
                                                   "resampler's length, so paste-back stays 1:1."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FACE_TRACK_DATA", "INT", "INT", "INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("face_clip", "track_data", "target_size", "enhanced_frames",
                    "num_runs", "frame_count", "enhanced", "report")
    FUNCTION = "crop"
    CATEGORY = "masking/face_gate"
    DESCRIPTION = ("Per-frame size-gated crop clip for a tracked face: only "
                   "frames where the face is below threshold are enhanced; "
                   "larger (zoomed-in) frames are left untouched. "
                   "target_size = native window * upscale_ratio.")

    def crop(self, images, mask_track, upscale_ratio, threshold_type, max_threshold_percent,
             hysteresis_percent, padding, smooth_alpha, max_size_deviation=0.5, size_smooth_alpha=0.4,
             min_threshold_percent=0.0, resampler="minimax_h3"):
        if mask_track.dim() == 2:
            mask_track = mask_track.unsqueeze(0)
        B, H, W, C = images.shape
        N = min(B, mask_track.shape[0])

        # Normalized threshold in [0,1] for the chosen measure, and a per-frame
        # measure in the SAME units, so the hysteresis state machine is identical
        # regardless of width/height/area.
        #   width  : face_bbox_width  / frame_width
        #   height : face_bbox_height / frame_height
        #   area   : (face_bbox_w * face_bbox_h) / (frame_w * frame_h)
        # Inputs are PERCENTS (0-100, same unit as the size report); convert to
        # fractions (0-1) here since the per-frame measure m (below) is a fraction.
        # threshold_type only selects which measure m the upper bound is compared to.
        thr = float(max_threshold_percent) / 100.0            # upper bound (fraction)
        thr_min = max(0.0, float(min_threshold_percent) / 100.0)  # lower bound (0 = off)
        hysteresis = float(hysteresis_percent) / 100.0        # dead-band (fraction)
        # Upper-bound dead-band (existing behavior).
        on_thresh = thr - hysteresis    # must be below this to (re)enable
        off_thresh = thr + hysteresis   # at/above this -> disable
        # Lower-bound dead-band (mirror of the upper one): enable only once the
        # face is clearly above the floor, disable when it drops clearly below.
        lo_on = thr_min + hysteresis    # must be above this to (re)enable
        lo_off = thr_min - hysteresis   # at/below this -> disable

        # Per-frame: box, and the normalized measure (None when face absent).
        boxes, measures = [], []
        for i in range(N):
            info = _frame_face_box((mask_track[i] > 0.5).cpu().numpy())
            boxes.append(info)
            if info is None:
                measures.append(None)
                continue
            fw = info[2] - info[0] + 1
            fh = info[3] - info[1] + 1
            if threshold_type == "height":
                m = fh / H
            elif threshold_type == "area":
                m = (fw * fh) / float(W * H)
            else:
                m = fw / W
            measures.append(m)

        # ── Observed face-size range across present frames, in ALL THREE measures.
        # This is a decision aid: run once, read the min–max for width/height/area,
        # then pick max_threshold_fraction / min_threshold_fraction from real data.
        w_fracs, h_fracs, a_fracs = [], [], []
        for info in boxes:
            if info is None:
                continue
            fw = info[2] - info[0] + 1
            fh = info[3] - info[1] + 1
            w_fracs.append(fw / W)
            h_fracs.append(fh / H)
            a_fracs.append((fw * fh) / float(W * H))

        def _rng(vals):
            return (min(vals), max(vals)) if vals else (0.0, 0.0)
        wlo, whi = _rng(w_fracs)
        hlo, hhi = _rng(h_fracs)
        alo, ahi = _rng(a_fracs)
        n_present = len(w_fracs)
        size_report = (
            f"faces on {n_present}/{N} frames — percent min–max: "
            f"width {wlo * 100:.1f}–{whi * 100:.1f}%, "
            f"height {hlo * 100:.1f}–{hhi * 100:.1f}%, "
            f"area {alo * 100:.2f}–{ahi * 100:.2f}%"
        )
        print(f"[FaceTrackCropAndGate] {size_report}. These are PERCENTS — set "
              f"max_threshold_percent just ABOVE the largest face you want "
              f"enhanced, and min_threshold_percent just BELOW the smallest, in "
              f"the threshold_type={threshold_type} column.")

        def _make_report(n_enh):
            return (size_report + f" | threshold_type={threshold_type}, "
                    f"max_threshold_percent={thr * 100:.2f}%, "
                    f"min_threshold_percent={thr_min * 100:.2f}% -> "
                    f"{n_enh} frame(s) enhanced")

        # Per-frame enhance decision with hysteresis. Absent frames are never
        # enhanced and do not change the on/off state (we hold it across gaps).
        enhance = [False] * N
        state_on = False
        for i in range(N):
            m = measures[i]
            if m is None:
                enhance[i] = False
                continue
            if state_on:
                # turn OFF if the face grows past the upper bound OR shrinks below
                # the lower bound (only checked when a lower bound is set).
                if m >= off_thresh or (thr_min > 0.0 and m <= lo_off):
                    state_on = False
            else:
                # turn ON only when below the upper bound AND above the lower bound.
                if m < on_thresh and (thr_min <= 0.0 or m > lo_on):
                    state_on = True
            enhance[i] = state_on

        enhanced_idx = [i for i in range(N) if enhance[i]]
        if not enhanced_idx:
            # No frame qualified for enhancement. Historically this raised and
            # killed the whole graph — but that defeats a workflow that WANTS to
            # skip the detailer branch (a large-face video needing no enhancement,
            # or a LazySwitchKJ gated on MaskHasFace). The gate runs eagerly
            # upstream, so a hard raise fires BEFORE any downstream switch can skip
            # it. Instead, emit a valid DUMMY no-op clip (mirrors FaceTrackSelectRun
            # for empty runs): a minimal batch on the resampler's grid, every entry
            # present=False, and n_real=0. Downstream Resize/H3/Decode run
            # harmlessly on it, and FaceTrackPasteBack skips every frame -> the
            # ORIGINAL video passes through unchanged. We still print a detailed
            # WARNING so a genuine misconfig stays visible in the log.
            present = [m for m in measures if m is not None]
            unit = {"height": "frame height", "area": "frame area"}.get(
                threshold_type, "frame width")
            if not present:
                why = ("the tracked mask was EMPTY on every frame — SAM3 tracked "
                       "nothing for the selected object index (check "
                       "SAM3_TrackToMask 'object_indices' and that the face is "
                       "actually detected)")
            else:
                minm, maxm = min(present), max(present)
                why = (f"no frame fell inside the enable window. Faces ranged "
                       f"{minm * 100:.1f}%–{maxm * 100:.1f}% of {unit}; a frame is "
                       f"enhanced only when the face is below {on_thresh * 100:.1f}% "
                       f"(threshold {thr * 100:.1f}% − hysteresis "
                       f"{hysteresis * 100:.1f}%)")
                if thr_min > 0.0:
                    why += (f" and above {lo_on * 100:.1f}% (min_threshold "
                            f"{thr_min * 100:.1f}% + hysteresis)")
                    if thr_min >= thr:
                        why += (f". NOTE: min_threshold ({thr_min * 100:.1f}%) ≥ "
                                f"threshold ({thr * 100:.1f}%) makes the enable "
                                f"window EMPTY, so no face can ever qualify")
                why += f" [threshold_type={threshold_type}]"
            print("[FaceTrackCropAndGate] WARNING: 0 frames selected for "
                  "enhancement — " + why + ". Emitting a no-op passthrough clip "
                  "(the original video is preserved on paste-back). Adjust "
                  "threshold / min_threshold / hysteresis if you expected "
                  "enhancement here.")
            side = 8
            dummy_len = _next_valid_clip_len(1, resampler)
            dummy = torch.zeros(dummy_len, side, side, C)
            data = {"entries": [{"frame": -1, "x0": 0, "y0": 0, "win": 0,
                                 "present": False, "run": -1, "cmask": None}
                                for _ in range(dummy_len)],
                    "orig_shape": (B, H, W, C), "out_side": side,
                    "upscale_ratio": float(upscale_ratio), "n_real": 0,
                    "n_runs": 0, "clip_length": dummy_len, "resampler": resampler,
                    "threshold_type": threshold_type, "max_threshold_frac": thr,
                    "min_threshold_frac": thr_min,
                    "ltx_length": dummy_len}
            target_size = max(8, int(round(side * upscale_ratio)))
            # frame_count = the (padded) clip length, so the resampler's `length`
            # can be driven directly without a separate GetImageSizeAndCount.
            # enhanced = False: no frame qualified. Wire this to LazySwitchKJ.switch
            # so the whole enhance branch (Resize/H3/sampler) is SKIPPED — the dummy
            # never reaches a resize node (which would collapse to 0 under
            # divisible_by) and the original video passes through on_false.
            return (dummy, data, target_size, 0, 0, dummy_len, False, _make_report(0))

        # ── Crop each enhanced frame TIGHT to its own face bbox + padding ──────
        # Earlier versions used one constant SQUARE window sized to the max face
        # over the whole clip, which (a) was far wider than tall faces, (b) was
        # inflated by the single largest/outlier frame, and (c) could clamp to the
        # full frame height — all making crops much larger than the mask. Instead
        # we crop tight per frame (preserving the face's own aspect via a square
        # box just big enough to contain that frame's face + padding) and resize
        # every crop to a common output size. LTX still gets a constant clip
        # dimension (the common size); each crop stays tight to its face.
        #
        # The common output size is driven by the MEDIAN enhanced-face size (not
        # max), so a single bloomed mask frame can't blow up the whole clip; that
        # size is only the resize target, not the crop extent, so no face is cut.
        raw_sides = []  # per-frame tight side BEFORE outlier clamp
        for i in enhanced_idx:
            b = boxes[i]
            fw = b[2] - b[0] + 1
            fh = b[3] - b[1] + 1
            raw_sides.append(int(round(max(fw, fh) * (1.0 + padding))))

        # Median tight side across enhanced frames — robust reference for "normal"
        # face size. (_frame_face_box already isolates the largest mask component,
        # so a face box won't be stretched by stray pixels; but on a few frames the
        # face component itself can be mis-sized, or a body component can dominate.
        # Clamp every frame's side to the median band so NO frame's crop can
        # suddenly engulf the body. This is the fix for "5 of 66 frames larger".)
        med = sorted(raw_sides)[len(raw_sides) // 2]
        # Clamp band from max_size_deviation: [med/(1+dev), med*(1+dev)].
        # dev <= 0 disables (lo=hi=med); very large dev effectively disables the cap.
        dev = max(0.0, float(max_size_deviation))
        if dev == 0.0:
            lo = hi = max(8, med)
        else:
            lo = max(8, int(round(med / (1.0 + dev))))
            hi = int(round(med * (1.0 + dev)))
        clamp_cap = min(H, W)

        per_frame = []  # (i, cx, cy, side)
        n_clamped = 0
        for k, i in enumerate(enhanced_idx):
            b = boxes[i]
            side = raw_sides[k]
            clamped = max(lo, min(side, hi))
            clamped = max(8, min(clamped, clamp_cap))
            if clamped != side:
                n_clamped += 1
            per_frame.append([i, b[4], b[5], clamped])

        # Common output size = clamped median (all frames resize to this).
        out_side = max(8, min(med, clamp_cap))
        if n_clamped:
            print(f"[FaceTrackCropAndGate] clamped {n_clamped}/{len(enhanced_idx)} "
                  f"outlier frame crop boxes to the median band "
                  f"[{lo},{hi}]px (median={med}px). These frames had a tall/merged "
                  f"mask (face+neck/body) that would otherwise engulf the body.")

        # Smooth centers across enhanced frames; reset at the start of each run.
        # A "run" is a maximal block of consecutive source-frame indices. With a
        # small->large->small subject the enhanced frames split into multiple runs
        # (e.g. [0..9] and [20..29]); we label each entry with its run index so a
        # downstream node could split per-run if desired, and so we never smooth
        # the crop center ACROSS a discontinuity.
        # Both the crop CENTER and the crop SIZE are temporally smoothed (EMA),
        # reset at the start of each run. Smoothing the SIZE is essential: the raw
        # per-frame mask bbox jitters a few px every frame, so an unsmoothed crop
        # side makes the zoom factor (out_side / side) change frame-to-frame, which
        # shows up as the face "breathing"/wobbling in scale — worst at padding=0
        # where no margin absorbs the jitter. Smoothing makes the crop size (and
        # thus the zoom) vary only gradually, following genuine size changes while
        # rejecting per-frame mask noise.
        clip, entries = [], []
        last_center = None
        last_side = None
        prev_i = None
        run_idx = -1
        for (i, cx, cy, side) in per_frame:
            new_run = last_center is None or (prev_i is not None and i != prev_i + 1)
            if new_run:
                run_idx += 1
                sm = (cx, cy)        # new run -> snap (no smoothing across the gap)
                sm_side = float(side)
            else:
                sm = (smooth_alpha * cx + (1 - smooth_alpha) * last_center[0],
                      smooth_alpha * cy + (1 - smooth_alpha) * last_center[1])
                sm_side = size_smooth_alpha * side + (1 - size_smooth_alpha) * last_side
            last_center = sm
            last_side = sm_side
            prev_i = i

            s = max(8, min(int(round(sm_side)), clamp_cap))
            half = s // 2
            x0 = int(round(sm[0] - half)); y0 = int(round(sm[1] - half))
            x0 = max(0, min(x0, W - s)); y0 = max(0, min(y0, H - s))
            crop = images[i, y0:y0 + s, x0:x0 + s, :]  # smoothed-size box around face
            # Crop the FACE MASK to the same box (kept at native s×s — small/cheap).
            # paste-back uses it as a face-shaped, feathered alpha so only face
            # pixels are written back (no rectangular seam / background overwrite).
            cmask = (mask_track[i, y0:y0 + s, x0:x0 + s] > 0.5).to(torch.float32).cpu()
            # Resize this crop to the common output size for a uniform clip
            # (area when shrinking, bicubic when enlarging — see _resize_hwc).
            crop = _resize_hwc(crop, out_side, out_side)
            clip.append(crop)
            # Store the SOURCE crop box (x0,y0,s) so paste-back lands exactly where
            # this frame's crop was, at its original scale. 'run' marks which
            # contiguous block this frame belongs to. 'present'=True means a real
            # enhanced frame (vs a pad frame appended below). 'cmask' is the
            # face-shaped alpha for that box.
            # Source face height (px) of THIS frame's face component (informational).
            # measure_frac = the face size in the gate's threshold_type units
            # (same fraction the gate compared against the window) — H3FaceRefine's
            # per-frame denoise ramps between min_threshold and max_threshold using it.
            _fb = boxes[i]
            face_px = float(_fb[3] - _fb[1] + 1)
            entries.append({"frame": i, "x0": x0, "y0": y0, "win": s,
                            "present": True, "run": run_idx, "cmask": cmask,
                            "face_px": face_px, "measure_frac": float(measures[i])})

        n_real = len(clip)
        n_runs = run_idx + 1

        # ── Pad the clip to a VALID resampler length ───────────────────────────
        # The resampler decodes back to a fixed grid: LTXVImgToVideo -> 8n+1
        # ((length-1)//8)+1 latent frames -> (T-1)*8+1); MiniMax H3 -> 17k+5. If
        # the clip length isn't on that grid, the decoded batch size DIFFERS from
        # the input, which would misalign processed_clip against our per-frame
        # entries (silent corruption / dropped frames). We pad the clip up to the
        # next valid length by repeating the last frame, and append matching 'pad'
        # entries (present=False) so paste-back ignores them. The workflow wires
        # the resampler's length from GetImageSizeAndCount on THIS clip, so it
        # automatically gets a valid length and returns the same frame count.
        target_len = _next_valid_clip_len(n_real, resampler)
        pad = target_len - n_real
        if pad > 0:
            last = clip[-1]
            for _ in range(pad):
                clip.append(last.clone())
                # Pad entries are non-present sentinels; paste-back skips them.
                entries.append({"frame": -1, "x0": 0, "y0": 0, "win": 0,
                                "present": False, "run": -1, "cmask": None})

        face_clip = torch.stack(clip, dim=0)  # [target_len, out_side, out_side, C]
        target_size = max(8, int(round(out_side * upscale_ratio)))
        grid = "17k+5" if resampler == "minimax_h3" else "8n+1"
        data = {"entries": entries, "orig_shape": (B, H, W, C), "out_side": out_side,
                "upscale_ratio": float(upscale_ratio), "n_real": n_real,
                "n_runs": n_runs, "clip_length": target_len, "resampler": resampler,
                # Gate window (fractions) + measure, so H3FaceRefine can ramp its
                # per-frame denoise between min_threshold and max_threshold.
                "threshold_type": threshold_type, "max_threshold_frac": thr,
                "min_threshold_frac": thr_min,
                # back-compat: older nodes/workflows read "ltx_length".
                "ltx_length": target_len}
        if n_runs > 1:
            print(f"[FaceTrackCropAndGate] {n_real} enhanced frames in {n_runs} "
                  f"separate runs (face crossed the threshold multiple times). "
                  f"They are concatenated into ONE clip of length {target_len} "
                  f"(padded by {pad}). A single {resampler} pass will bridge the run "
                  f"boundary; for strict per-run temporal coherence, process each "
                  f"run as its own pass. Paste-back maps every frame correctly "
                  f"regardless.")
        elif pad > 0:
            print(f"[FaceTrackCropAndGate] padded clip {n_real}->{target_len} "
                  f"frames to a valid {resampler} length ({grid}); pad frames are "
                  f"ignored on paste-back.")
        # frame_count = the (padded) clip length; feed the resampler's `length`
        # from this directly instead of a separate GetImageSizeAndCount node.
        # enhanced = True: >=1 frame qualified, so the enhance branch should run.
        return (face_clip, data, target_size, n_real, n_runs, target_len,
                n_real > 0, _make_report(n_real))


class FaceTrackSelectRun:
    """Select ONE contiguous run from a FaceTrackCropAndGate clip.

    For the per-run-pass workflow (small->large->small etc.): instead of sending
    the whole multi-run clip through one LTX pass (which bridges run boundaries),
    instantiate one of these per run index, give each its own LTX branch, then
    chain FaceTrackPasteBack nodes. Each output is a self-contained clip +
    track_data limited to that run, padded to a valid LTX length on its own — so
    every existing downstream node (Resize, GetImageSizeAndCount, LTX, PasteBack)
    works unchanged on a single run.

    run_index beyond the number of runs yields an empty result (paste-back no-op),
    so a fixed set of N FaceTrackSelectRun nodes safely covers clips with <= N
    runs. enhanced_frames=0 signals "this run doesn't exist".
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_clip": ("IMAGE",),
                "track_data": ("FACE_TRACK_DATA",),
                "run_index": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                              "tooltip": "Which contiguous run to extract (0 = first). "
                                         "Out-of-range yields an empty clip (no-op)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FACE_TRACK_DATA", "INT", "INT", "INT")
    RETURN_NAMES = ("run_clip", "run_track_data", "target_size", "enhanced_frames",
                    "frame_count")
    FUNCTION = "select"
    CATEGORY = "masking/face_gate"
    DESCRIPTION = ("Extract one run from a multi-run face clip for an independent "
                   "LTX pass. Pad each run to its own valid LTX length. frame_count "
                   "is that padded length — wire it straight into the resampler's "
                   "length (no GetImageSizeAndCount needed).")

    def select(self, face_clip, track_data, run_index):
        entries = track_data.get("entries", [])
        out_side = track_data.get("out_side", face_clip.shape[1] if face_clip.shape[0] else 8)
        ratio = track_data.get("upscale_ratio", 2.0)
        resampler = track_data.get("resampler", "ltx")  # match the crop node's grid
        C = face_clip.shape[-1] if face_clip.ndim == 4 else 3

        # Indices (into face_clip / entries) of REAL frames in the requested run.
        sel = [k for k, e in enumerate(entries)
               if e.get("present", False) and e.get("run", -1) == run_index]
        if not sel:
            # Out-of-range / empty run. Return a DUMMY clip (not 0 frames): a
            # 0-frame batch crashes downstream Resize/resampler (torch.stack on
            # empty), which would make over-provisioned branches fail. The dummy
            # frames survive Resize->resampler->Decode harmlessly, and every entry
            # is present=False so paste-back skips it -> a true no-op branch. This
            # is what makes it safe to wire MORE run branches than a clip has.
            # Length honours the resampler's minimum grid value (ltx=1, h3=5) so
            # the dummy is itself a valid clip for the resampler.
            side = out_side if out_side and out_side > 0 else 8
            dummy_len = _next_valid_clip_len(1, resampler)
            dummy = torch.zeros(dummy_len, side, side, C)
            data = {"entries": [{"frame": -1, "x0": 0, "y0": 0, "win": 0,
                                 "present": False, "run": -1} for _ in range(dummy_len)],
                    "orig_shape": track_data.get("orig_shape"),
                    "out_side": side, "upscale_ratio": float(ratio),
                    "n_real": 0, "n_runs": 0, "clip_length": dummy_len,
                    "resampler": resampler, "ltx_length": dummy_len}
            return (dummy, data, max(8, int(round(side * ratio))), 0, dummy_len)

        run_frames = [face_clip[k] for k in sel]
        run_entries = [dict(entries[k]) for k in sel]  # copy; keep frame/x0/y0/win
        n_real = len(run_frames)

        # Pad THIS run to its own valid resampler length (same grid as the crop).
        target_len = _next_valid_clip_len(n_real, resampler)
        pad = target_len - n_real
        if pad > 0:
            last = run_frames[-1]
            for _ in range(pad):
                run_frames.append(last.clone())
                run_entries.append({"frame": -1, "x0": 0, "y0": 0, "win": 0,
                                    "present": False, "run": run_index})

        run_clip = torch.stack(run_frames, dim=0)
        data = {"entries": run_entries, "orig_shape": track_data.get("orig_shape"),
                "out_side": out_side, "upscale_ratio": float(ratio),
                "n_real": n_real, "n_runs": 1, "clip_length": target_len,
                "resampler": resampler, "ltx_length": target_len}
        target_size = max(8, int(round(out_side * ratio)))
        return (run_clip, data, target_size, n_real, target_len)


class FaceTrackRunCount:
    """Report how many separate runs a FaceTrackCropAndGate clip contains.

    Because run count is only known at execution time (it depends on SAM3 +
    the gate), you cannot know it before running. Run the workflow once with
    this node connected: it prints/returns num_runs so you know how many
    FaceTrackSelectRun branches (= LTX passes) you actually need. Provision that
    many branches; any extra branches are safe no-ops (empty runs pass through).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"track_data": ("FACE_TRACK_DATA",)}}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("num_runs", "enhanced_frames")
    FUNCTION = "count"
    CATEGORY = "masking/face_gate"
    DESCRIPTION = ("Number of separate runs (and total enhanced frames) in a "
                   "tracked face clip. Run once to learn how many LTX passes the "
                   "per-run workflow needs.")

    def count(self, track_data):
        entries = track_data.get("entries", [])
        runs = sorted({e.get("run", -1) for e in entries
                       if e.get("present", False) and e.get("run", -1) >= 0})
        num_runs = len(runs)
        n_real = int(track_data.get("n_real", sum(1 for e in entries if e.get("present", False))))
        print(f"[FaceTrackRunCount] num_runs={num_runs} (run indices {runs}), "
              f"enhanced_frames={n_real}. Provision {num_runs} FaceTrackSelectRun "
              f"branch(es) (run_index 0..{max(0, num_runs - 1)}); extra branches "
              f"are harmless no-ops.")
        return (num_runs, n_real)


class FaceTrackPasteBack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "processed_clip": ("IMAGE",),     # [N, *, *, C] enhanced face clip (any size; resized to window)
                "track_data": ("FACE_TRACK_DATA",),
                "feather": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01,
                                      "tooltip": "Edge softness. In 'mask' mode: Gaussian-blur radius of the "
                                                 "face-shaped alpha, as a fraction of the crop side. In 'rectangle' "
                                                 "mode: width of the linear edge ramp."}),
                "blend_mode": (["mask", "rectangle"], {"default": "mask",
                                        "tooltip": "'mask' composites using the FACE-SHAPED segmentation alpha "
                                                   "(only face pixels are written; no rectangular seam — recommended). "
                                                   "'rectangle' is the legacy feathered-square blend."}),
                "only_present_frames": ("BOOLEAN", {"default": True,
                                        "tooltip": "Only composite frames where the face was actually detected."}),
                "colour_match": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "Match the processed face's per-channel mean/std to the "
                                                   "original region it replaces, blended by this amount. The "
                                                   "crop and the frame went through independent resample passes, "
                                                   "so the refined face can come back subtly brighter/shifted and "
                                                   "read as pasted-on; matching removes that seam. 0 = off "
                                                   "(unchanged legacy behaviour); 1 = full match. Helpful with "
                                                   "generative resamplers like MiniMax H3."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "paste"
    CATEGORY = "masking/face_gate"
    DESCRIPTION = ("Resize the processed face clip back to its native window "
                   "(undoing upscale_ratio) and composite per frame.")

    def paste(self, original_images, processed_clip, track_data, feather,
              blend_mode="mask", only_present_frames=True, colour_match=0.0):
        out = original_images.clone()
        entries = track_data.get("entries", [])
        if len(entries) == 0:
            # Nothing was enhanced (or upstream produced no data) -> pass through.
            return (out,)
        M = processed_clip.shape[0]

        # ── Strict count guard ────────────────────────────────────────────────
        # processed_clip MUST line up 1:1 with entries by index (processed_clip[k]
        # is the enhanced version of entries[k]). The crop node pads its clip to a
        # valid LTX length (8n+1) so a correctly-wired LTX pass returns exactly
        # that many frames. If the counts differ, something between crop and here
        # changed the batch size (wrong LTX 'length', a reorder/dedupe node, a
        # frame-rate convert, etc.). Silently truncating would paste enhanced
        # faces onto the WRONG frames, so refuse with a precise message instead.
        expected = track_data.get("clip_length", track_data.get("ltx_length", len(entries)))
        resampler = track_data.get("resampler", "ltx")
        length_input = ("LTXVImgToVideo.length" if resampler == "ltx"
                        else "the MiniMax H3 node's length")
        if M != len(entries):
            raise ValueError(
                f"FaceTrackPasteBack: processed_clip has {M} frames but track_data "
                f"describes {len(entries)} (expected {expected}). They must match "
                f"1:1 — processed_clip[k] is the enhanced entries[k]. Most likely "
                f"the resampler 'length' is not wired from this clip's frame count "
                f"(use GetImageSizeAndCount on the upscaled face_clip -> "
                f"{length_input}), or a node between the crop and here "
                f"changed the batch size/order. Fix the wiring rather than letting "
                f"faces paste onto wrong frames."
            )

        for idx, e in enumerate(entries):
            if only_present_frames and not e.get("present", True):
                continue  # pad / absent frame -> skip (correct, not truncation)
            f = e["frame"]; x0 = e["x0"]; y0 = e["y0"]; win = e["win"]
            if win <= 0 or f < 0:
                continue
            # Resize the processed face back to its native box (area when
            # shrinking — alias-free downsample — bicubic when enlarging).
            face_r = _resize_hwc(processed_clip[idx], win, win).to(out.dtype).to(out.device)

            # Optional colour transfer: match the refined face's per-channel
            # mean/std to the original crop region so an independent resample
            # pass (e.g. H3) doesn't paste back a subtly brighter/shifted face.
            if colour_match > 0.0:
                reg0 = out[f, y0:y0 + win, x0:x0 + win, :]
                if reg0.shape[0] == win and reg0.shape[1] == win:
                    fm = face_r.reshape(-1, face_r.shape[-1])
                    rm = reg0.reshape(-1, reg0.shape[-1])
                    f_mean, f_std = fm.mean(0), fm.std(0) + 1e-5
                    r_mean, r_std = rm.mean(0), rm.std(0) + 1e-5
                    matched = (face_r - f_mean) / f_std * r_std + r_mean
                    face_r = (face_r + colour_match * (matched - face_r)).clamp(0.0, 1.0)

            cmask = e.get("cmask")
            if blend_mode == "mask" and cmask is not None and cmask.numel() > 0:
                # Face-SHAPED alpha: resize the stored face mask to the box and
                # Gaussian-feather its edge, so only face pixels are written and
                # the surrounding background is left untouched (no square seam).
                a = cmask.to(out.dtype).to(out.device)
                if a.shape[0] != win or a.shape[1] != win:
                    a = F.interpolate(a.unsqueeze(0).unsqueeze(0), size=(win, win),
                                      mode="bilinear", align_corners=False)[0, 0]
                fpx = int(win * feather)
                a = _gaussian_blur_2d(a, fpx)
                alpha = a.unsqueeze(-1)
            else:
                # Legacy rectangular blend: full square with a linear edge ramp.
                fpx = max(1, int(win * feather))
                alpha = torch.ones(win, win, dtype=out.dtype, device=out.device)
                if fpx > 0:
                    ramp = torch.linspace(0, 1, steps=fpx, dtype=out.dtype, device=out.device)
                    alpha[:fpx, :] *= ramp.view(-1, 1)
                    alpha[-fpx:, :] *= ramp.flip(0).view(-1, 1)
                    alpha[:, :fpx] *= ramp.view(1, -1)
                    alpha[:, -fpx:] *= ramp.flip(0).view(1, -1)
                alpha = alpha.unsqueeze(-1)

            region = out[f, y0:y0 + win, x0:x0 + win, :]
            # Guard against edge size mismatch
            if region.shape[0] != win or region.shape[1] != win:
                rh, rw = region.shape[0], region.shape[1]
                face_r = face_r[:rh, :rw, :]
                alpha = alpha[:rh, :rw, :]
            out[f, y0:y0 + region.shape[0], x0:x0 + region.shape[1], :] = \
                region * (1 - alpha) + face_r * alpha
        return (out,)


class MaskHasFace:
    """MASK -> BOOLEAN: True if any frame's mask has a face region.

    Drives a LazySwitchKJ so the whole crop -> upscale -> LTX -> paste branch is
    skipped (never executed) when SAM3 tracked nothing. Must sit OUTSIDE the
    gated branch (it reads the SAM3 mask directly, not the crop output).

    `min_pixels` guards against a few stray mask pixels counting as a face.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "min_pixels": ("INT", {"default": 1, "min": 1, "max": 1_000_000, "step": 1,
                                       "tooltip": "A frame counts as having a face only if its mask has at "
                                                  "least this many set pixels (guards against stray specks)."}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "INT")
    RETURN_NAMES = ("has_face", "frames_with_face")
    FUNCTION = "check"
    CATEGORY = "masking/face_gate"
    DESCRIPTION = ("True if any frame's mask contains a face (>= min_pixels set). "
                   "Wire into LazySwitchKJ.switch to skip the detailer branch "
                   "when no face is detected.")

    def check(self, masks, min_pixels=1):
        import torch
        m = masks
        if m.dim() == 2:
            m = m.unsqueeze(0)
        per_frame = (m > 0.5).flatten(1).sum(dim=1)   # set-pixels per frame
        frames = int((per_frame >= min_pixels).sum().item())
        has = frames > 0
        print(f"[MaskHasFace] frames_with_face={frames}/{m.shape[0]} -> has_face={has}")
        return (has, frames)


# ─────────────────────────────────────────────────────────────────────────────
# MiniMax H3 face-refine (one node: img2img inject + lipsync + per-frame denoise)
# ─────────────────────────────────────────────────────────────────────────────
# MiniMax H3 is a joint audio-video latent-diffusion model. Its stock nodes always
# build a ZEROS latent (references are conditioning re-injected each step), so there
# is no stock video-to-video path. H3FaceRefine wraps the three steps needed to use
# H3 as the face resampler, in the one correct order, so the workflow needs a single
# node instead of three:
#   1. inject the real (upscaled) face clip into the AV latent's VIDEO stream, turning
#      SamplerCustomAdvanced + truncated sigmas into genuine img2img (output tracks the
#      input frames 1:1);
#   2. lock the ORIGINAL audio into the AUDIO stream and mask sampling so only video
#      denoises while attending to that fixed audio — this drives lip-sync;
#   3. scale denoise along time by face size (strong on tiny faces that must be
#      synthesised, gentle on large faces with real detail) via the latent noise mask.
# Feed its outputs straight into BasicGuider/BasicScheduler (model) and
# SamplerCustomAdvanced (av_latent). audio/audio_vae are optional — omit to skip
# lip-sync. Adapted from ComfyUI-H3-FaceRefine (Carasibana, MIT) and
# ComfyUI-H3-NativeAudioLock (Shrek3OnVH5). comfy / torchaudio are imported lazily so
# nodes.py still imports (and unit-tests still run) without a full ComfyUI install.


class H3FaceRefine:
    """MiniMax H3 face-refine setup in one node: img2img + lipsync + per-frame denoise.

    Wire the H3 AV latent (from MiniMaxH3ReferenceToVideo) into `av_latent`, the
    upscaled face clip into `images`, the H3 VIDEO vae into `vae`, the crop node's
    `track_data`, and (optionally) the source `audio` + `audio_vae` for lip-sync.
    Outputs the patched `model` (-> BasicGuider / BasicScheduler) and the prepared
    `av_latent` (-> SamplerCustomAdvanced.latent_image). Set overall strength with
    BasicScheduler's `denoise` (NOT SplitSigmas).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "av_latent": ("LATENT",),
                "images": ("IMAGE",),
                "vae": ("VAE",),
                "track_data": ("FACE_TRACK_DATA",),
                "strength_small_face": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Denoise multiplier where the face is SMALLEST — i.e. at the gate's "
                               "min_threshold_percent. 1.0 = the full denoise set on BasicScheduler."}),
                "strength_large_face": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Denoise multiplier where the face is LARGEST — i.e. at the gate's "
                               "max_threshold_percent. Lower preserves detail (and keeps edges matching)."}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 4.0, "step": 0.1,
                    "tooltip": ">1 keeps strength high until the face is genuinely large; <1 drops early."}),
                "smooth_frames": ("INT", {"default": 9, "min": 1, "max": 61, "step": 2,
                    "tooltip": "Smooth the strength curve over time; abrupt denoise changes read as a pop."}),
            },
            "optional": {
                "audio_vae": ("VAE", {"tooltip": "H3 AUDIO vae. Needed with `audio` for lip-sync."}),
                "audio": ("AUDIO", {"tooltip": "Source audio (or isolated vocals). Omit to skip lip-sync."}),
            },
        }

    RETURN_TYPES = ("MODEL", "LATENT", "STRING")
    RETURN_NAMES = ("model", "av_latent", "report")
    FUNCTION = "run"
    CATEGORY = "masking/face_gate"
    DESCRIPTION = ("MiniMax H3 face-refine setup: img2img inject + audio lip-sync + "
                   "per-frame denoise, in one node. Per-frame denoise ramps between "
                   "strength_small_face (at the gate's min_threshold_percent) and "
                   "strength_large_face (at max_threshold_percent), using the gate's "
                   "threshold_type measure.")

    def run(self, model, av_latent, images, vae, track_data,
            strength_small_face, strength_large_face,
            gamma, smooth_frames, audio_vae=None, audio=None):
        import comfy.nested_tensor
        reports = []

        # ── 1. inject the face clip into the VIDEO stream (img2img) ──────────────
        samples = av_latent.get("samples")
        if samples is None:
            raise KeyError('LATENT is missing "samples".')
        if not (isinstance(samples, comfy.nested_tensor.NestedTensor)
                or getattr(samples, "is_nested", False)):
            raise ValueError("Expected a MiniMax H3 joint AV latent (NestedTensor). Feed the "
                             "LATENT output of MiniMaxH3ReferenceToVideo / EmptyMiniMaxH3LatentAV.")
        members = list(samples.unbind())
        video_tmpl = members[0]
        encoded = vae.encode(images[..., :3])
        if encoded.ndim == 4:  # [B,C,H,W] -> [1,C,T,H,W]
            encoded = encoded.unsqueeze(0).movedim(1, 2)
        tgt_t, tgt_h, tgt_w = video_tmpl.shape[-3], video_tmpl.shape[-2], video_tmpl.shape[-1]
        got_t, got_h, got_w = encoded.shape[-3], encoded.shape[-2], encoded.shape[-1]
        if (got_h, got_w) != (tgt_h, tgt_w):
            raise ValueError(f"Spatial latent mismatch: encoded {got_h}x{got_w} but the AV latent "
                             f"expects {tgt_h}x{tgt_w}. The crop canvas and the H3 node's width/height "
                             f"must match (both are pixels/16).")
        if got_t != tgt_t:
            # crop node with resampler='minimax_h3' pads to 17k+5, so this is a guard.
            if got_t > tgt_t:
                encoded = encoded[..., :tgt_t, :, :]
            else:
                padt = video_tmpl[..., : tgt_t - got_t, :, :].to(encoded.device, encoded.dtype)
                encoded = torch.cat([encoded, padt], dim=-3)
            reports.append(f"WARNING temporal mismatch enc t={got_t} vs latent t={tgt_t} "
                           f"(set crop resampler='minimax_h3')")
        video_latent = encoded.to(video_tmpl.device, video_tmpl.dtype)
        members[0] = video_latent
        cur = dict(av_latent)
        cur["samples"] = comfy.nested_tensor.NestedTensor(tuple(members))
        reports.append(f"inject video latent {tuple(video_latent.shape)}")

        # ── 2. lock the ORIGINAL audio (lip-sync), denoise video only ───────────
        out_model = model
        if audio is not None and audio_vae is not None:
            import torchaudio
            if len(members) < 2:
                raise ValueError("AV latent has no audio stream to lock.")
            audio_tmpl = members[1]
            waveform = audio["waveform"][:1]
            sr = int(audio["sample_rate"])
            vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
            if sr != vae_rate:
                waveform = torchaudio.functional.resample(waveform, sr, vae_rate)
            exact_audio = audio_vae.encode(waveform.movedim(1, -1))
            at = audio_tmpl.shape[-1]
            if exact_audio.shape[-1] > at:
                exact_audio = exact_audio[..., :at]
            elif exact_audio.shape[-1] < at:
                exact_audio = F.pad(exact_audio, (0, at - exact_audio.shape[-1]))
            cur["samples"] = comfy.nested_tensor.NestedTensor((video_latent, exact_audio))
            cur["noise_mask"] = comfy.nested_tensor.NestedTensor(
                (torch.ones_like(video_latent), torch.zeros_like(exact_audio)))
            out_model = model.clone()
            topts = out_model.model_options["transformer_options"] = (
                out_model.model_options.get("transformer_options", {}).copy())
            topts["minimax_h3_lock_audio_clean"] = True
            reports.append("lipsync: locked audio, video-only denoise")
        else:
            reports.append("lipsync: skipped (no audio)")

        # ── 3. per-frame denoise via the video noise mask ───────────────────────
        latent_t = video_latent.shape[-3]
        entries = track_data.get("entries", [])
        if not entries:
            raise ValueError("track_data has no entries.")
        # Per-frame face measure in the GATE's threshold_type units (a fraction),
        # recorded by FaceTrackCropAndGate. The denoise ramp bounds come straight
        # from the gate: min_threshold (smallest qualifying face) -> full denoise,
        # max_threshold (largest) -> gentle. So the ramp automatically follows
        # threshold_type and the exact window you set on the gate.
        meas_list, last = [], None
        for e in entries:
            mf = e.get("measure_frac")
            if mf is None or mf <= 0:
                mf = last
            if mf is not None:
                last = mf
            meas_list.append(mf)
        first_valid = next((m for m in meas_list if m is not None), 0.0)
        meas = np.array([first_valid if m is None else m for m in meas_list],
                        dtype=np.float64)
        lo = track_data.get("min_threshold_frac")
        hi = track_data.get("max_threshold_frac")
        if lo is None or hi is None or float(hi) - float(lo) < 1e-9:
            # Fallback (e.g. track_data from an older gate): normalise to this
            # clip's own observed face-size range instead of the gate window.
            lo, hi = float(meas.min()), float(meas.max())
            reports.append("denoise ramp: clip-relative (gate thresholds absent)")
        else:
            lo, hi = float(lo), float(hi)
            reports.append(f"denoise ramp: {lo * 100:.2f}%–{hi * 100:.2f}% "
                           f"({track_data.get('threshold_type', 'width')})")
        if hi - lo < 1e-9:
            t = np.zeros_like(meas)
        else:
            t = np.clip((meas - lo) / (hi - lo), 0.0, 1.0)
        t = t ** float(gamma)
        strength = strength_small_face + (strength_large_face - strength_small_face) * t
        k = int(smooth_frames)
        if k > 1 and strength.size > 1:
            r = k // 2
            xs = np.arange(-r, r + 1)
            sig = max(k / 6.0, 0.5)
            ker = np.exp(-(xs ** 2) / (2 * sig * sig))
            ker /= ker.sum()
            strength = np.convolve(np.pad(strength, r, mode="edge"), ker, mode="valid")
        strength = np.clip(strength, 0.0, 1.0)
        s = torch.from_numpy(strength).float().view(1, 1, -1)
        s = F.interpolate(s, size=int(latent_t), mode="linear", align_corners=True)
        s = s.view(1, 1, int(latent_t), 1, 1).to(video_latent.device, torch.float32)
        vmask = s.expand(video_latent.shape[0], video_latent.shape[1], latent_t,
                         video_latent.shape[-2], video_latent.shape[-1]).contiguous()
        prev = cur.get("noise_mask")
        if prev is not None and (isinstance(prev, comfy.nested_tensor.NestedTensor)
                                 or getattr(prev, "is_nested", False)):
            pm = list(prev.unbind())          # keep the audio side (zeros) intact
            pm[0] = vmask.to(pm[0].dtype)
            cur["noise_mask"] = comfy.nested_tensor.NestedTensor(tuple(pm))
        else:
            mem2 = list(cur["samples"].unbind())
            audio_zero = torch.zeros_like(mem2[1]) if len(mem2) > 1 else None
            cur["noise_mask"] = comfy.nested_tensor.NestedTensor(
                (vmask.to(video_latent.dtype),) + ((audio_zero,) if audio_zero is not None else ()))
        reports.append(f"denoise {strength.max():.2f}..{strength.min():.2f} over {len(strength)} "
                       f"frames ({latent_t} latent)")

        report = "[H3FaceRefine] " + " | ".join(reports)
        print(report)
        return (out_model, cur, report)


class FaceTrackAudioSlice:
    """Reindex source audio to a gated face clip's frames — for correct H3 lip-sync.

    FaceTrackCropAndGate keeps only the small-face frames (often several
    non-contiguous runs) and concatenates them, so the clip fed to H3 is a SUBSET
    of the timeline in clip order — NOT the whole video. H3's audio lock aligns
    audio to video 1:1, so feeding the whole track would sync the generated mouths
    to the wrong words. This builds the matching audio: for each clip frame it
    takes a 1/target_fps-second window from the source at that frame's ORIGINAL
    time (source frame index ÷ source_fps), and concatenates them in clip order
    (silence for pad/absent frames). The result is target_fps-framed audio that
    lines up 1:1 with the clip — wire it into H3FaceRefine's `audio` (and the H3
    reference node's `ref_audio`). Mux the FULL original audio at the save node.

    For a 24fps source and one contiguous run this is just a clean slice; for
    other fps / multiple runs it resamples per frame onto H3's 24fps grid.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "track_data": ("FACE_TRACK_DATA",),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 240.0, "step": 0.001,
                    "tooltip": "FPS of the SOURCE video (wire VHS_VideoInfo source_fps). Maps each "
                               "gated frame's index to its time in the original audio."}),
                "target_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0,
                    "tooltip": "FPS H3 interprets the clip at (24). Each clip frame gets a "
                               "1/target_fps-second audio window."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    FUNCTION = "slice"
    CATEGORY = "masking/face_gate"
    DESCRIPTION = ("Reindex source audio to a gated face clip's frames so H3 lip-sync "
                   "matches (per clip frame: a window from the source at that frame's "
                   "original time; silence for pad frames).")

    def slice(self, audio, track_data, source_fps=24.0, target_fps=24.0):
        wav = audio.get("waveform")
        sr = int(audio.get("sample_rate", 0))
        entries = track_data.get("entries", [])
        if wav is None or sr <= 0 or not entries:
            return (audio, "[FaceTrackAudioSlice] passthrough (missing waveform/sr/entries)")
        if source_fps <= 0:
            source_fps = 24.0
        S = wav.shape[-1]
        win = max(1, int(round(sr / float(target_fps))))   # samples per output frame
        lead = wav.shape[:-1]                                # [B, C]
        segs, n_present, n_pad = [], 0, 0
        for e in entries:
            if e.get("present", False) and e.get("frame", -1) >= 0:
                s0 = int(round((e["frame"] / float(source_fps)) * sr))
                s0 = max(0, min(s0, S))
                seg = wav[..., s0:s0 + win]
                if seg.shape[-1] < win:                      # zero-pad a short tail
                    pad = torch.zeros(*lead, win - seg.shape[-1], dtype=wav.dtype, device=wav.device)
                    seg = torch.cat([seg, pad], dim=-1)
                segs.append(seg)
                n_present += 1
            else:
                segs.append(torch.zeros(*lead, win, dtype=wav.dtype, device=wav.device))
                n_pad += 1
        out = torch.cat(segs, dim=-1)
        report = (f"[FaceTrackAudioSlice] {len(entries)} clip frames "
                  f"({n_present} present, {n_pad} pad) -> {out.shape[-1]} samples @ {sr}Hz "
                  f"({out.shape[-1] / sr:.2f}s) src_fps={source_fps} target_fps={target_fps}")
        print(report)
        return ({"waveform": out, "sample_rate": sr}, report)


NODE_CLASS_MAPPINGS = {
    "FaceSizeGateMask": FaceSizeGateMask,
    "FaceCropAndGate": FaceCropAndGate,
    "FacePasteBack": FacePasteBack,
    "FaceTrackCropAndGate": FaceTrackCropAndGate,
    "FaceTrackSelectRun": FaceTrackSelectRun,
    "FaceTrackRunCount": FaceTrackRunCount,
    "FaceTrackPasteBack": FaceTrackPasteBack,
    "MaskHasFace": MaskHasFace,
    "H3FaceRefine": H3FaceRefine,
    "FaceTrackAudioSlice": FaceTrackAudioSlice,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceSizeGateMask": "Face Size Gate (Mask)",
    "FaceCropAndGate": "Face Crop & Gate (per-face)",
    "FacePasteBack": "Face Paste Back (per-face)",
    "FaceTrackCropAndGate": "Face Track Crop & Gate (coherent)",
    "FaceTrackSelectRun": "Face Track Select Run (per-run)",
    "FaceTrackRunCount": "Face Track Run Count",
    "FaceTrackPasteBack": "Face Track Paste Back (coherent)",
    "MaskHasFace": "Mask Has Face (bool)",
    "H3FaceRefine": "MiniMax H3 Face Refine",
    "FaceTrackAudioSlice": "Face Track Audio Slice",
}
