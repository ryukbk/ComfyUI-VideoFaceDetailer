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
                                    "tooltip": "Which face dimension the size gate uses: 'width' -> "
                                               "max_width_fraction (fraction of frame width); 'height' -> "
                                               "max_height_fraction (fraction of frame height); 'area' -> "
                                               "max_area_percent (face bbox as a % of the whole frame area). "
                                               "Only the matching parameter below is used."}),
                "max_width_fraction": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.005,
                                                 "tooltip": "[threshold_type=width] Enhance a frame ONLY while the "
                                                            "face is narrower than this fraction of the frame width."}),
                "max_height_fraction": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.005,
                                                  "tooltip": "[threshold_type=height] Enhance a frame ONLY while the "
                                                             "face is shorter than this fraction of the frame height."}),
                "max_area_percent": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                             "tooltip": "[threshold_type=area] Enhance a frame ONLY while the face "
                                                        "bbox occupies less than this percent of the whole frame "
                                                        "area. E.g. 12.1 = faces smaller than 12.1% of the frame."}),
                "hysteresis": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005,
                                         "tooltip": "Dead-band around the threshold, in the SAME normalized units as "
                                                    "the chosen measure (fraction of width/height, or fraction of "
                                                    "area where 0.02 = 2% of frame area). Stops on/off flicker when "
                                                    "the face hovers at the boundary. ON below (threshold - "
                                                    "hysteresis), OFF at/above (threshold + hysteresis)."}),
                "padding": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05}),
                "smooth_alpha": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01,
                                           "tooltip": "Crop CENTER smoothing (EMA). 1.0 = follow the face exactly "
                                                      "(no positional lag); lower = steadier but lags fast moves. "
                                                      "Use 1.0 if you see the face drift within the crop on fast "
                                                      "motion."}),
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
            },
        }

    RETURN_TYPES = ("IMAGE", "FACE_TRACK_DATA", "INT", "INT", "INT")
    RETURN_NAMES = ("face_clip", "track_data", "target_size", "enhanced_frames", "num_runs")
    FUNCTION = "crop"
    CATEGORY = "masking/face_gate"
    DESCRIPTION = ("Per-frame size-gated crop clip for a tracked face: only "
                   "frames where the face is below threshold are enhanced; "
                   "larger (zoomed-in) frames are left untouched. "
                   "target_size = native window * upscale_ratio.")

    def crop(self, images, mask_track, upscale_ratio, threshold_type, max_width_fraction,
             max_height_fraction, max_area_percent, hysteresis,
             padding, smooth_alpha, max_size_deviation=0.5, size_smooth_alpha=0.4):
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
        if threshold_type == "height":
            thr = float(max_height_fraction)
        elif threshold_type == "area":
            thr = float(max_area_percent) / 100.0
        else:  # width (default)
            thr = float(max_width_fraction)
        on_thresh = thr - hysteresis    # must be below this to (re)enable
        off_thresh = thr + hysteresis   # at/above this -> disable

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
                if m >= off_thresh:
                    state_on = False
            else:
                if m < on_thresh:
                    state_on = True
            enhance[i] = state_on

        enhanced_idx = [i for i in range(N) if enhance[i]]
        if not enhanced_idx:
            # No frame qualified. Emitting an empty batch crashes downstream nodes
            # (Resize/LTX call torch.stack on it). Raise a clear, actionable error
            # instead, with the diagnostics needed to fix it.
            present = [m for m in measures if m is not None]
            unit = {"height": "frame height", "area": "frame area"}.get(
                threshold_type, "frame width")
            if not present:
                detail = ("the tracked mask was EMPTY on every frame — SAM3 "
                          "tracked nothing for the selected object index. Check "
                          "SAM3_TrackToMask 'object_indices' and that the face is "
                          "actually detected.")
            else:
                minm = min(present)
                detail = (f"the face never dropped below the enable threshold. "
                          f"Smallest face was {minm * 100:.1f}% of {unit}, but "
                          f"enhancement only turns on below (threshold - hysteresis) "
                          f"= {on_thresh * 100:.1f}% (threshold_type={threshold_type}). "
                          f"Raise the threshold, lower hysteresis, or accept that no "
                          f"face is small enough.")
            raise ValueError(
                "FaceTrackCropAndGate: 0 frames selected for enhancement — "
                + detail +
                " (If you want large-face videos to pass through unchanged, this "
                "enhancement branch cannot run on zero frames; bypass it or route "
                "the original video to the output.)"
            )

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
            # Resize this crop to the common output size for a uniform clip
            # (area when shrinking, bicubic when enlarging — see _resize_hwc).
            crop = _resize_hwc(crop, out_side, out_side)
            clip.append(crop)
            # Store the SOURCE crop box (x0,y0,s) so paste-back lands exactly where
            # this frame's crop was, at its original scale. 'run' marks which
            # contiguous block this frame belongs to. 'present'=True means a real
            # enhanced frame (vs a pad frame appended below).
            entries.append({"frame": i, "x0": x0, "y0": y0, "win": s,
                            "present": True, "run": run_idx})

        n_real = len(clip)
        n_runs = run_idx + 1

        # ── Pad the clip to a VALID LTX length (8n+1) ──────────────────────────
        # LTXVImgToVideo builds latent temporal size ((length-1)//8)+1 and decodes
        # back to (T-1)*8+1 frames. If the clip length isn't 8n+1, the decoded
        # batch size DIFFERS from the input, which would misalign processed_clip
        # against our per-frame entries (silent corruption / dropped frames). We
        # pad the clip up to the next 8n+1 by repeating the last frame, and append
        # matching 'pad' entries (present=False) so paste-back ignores them. The
        # workflow wires LTX length from GetImageSizeAndCount on THIS clip, so it
        # automatically gets a valid length and returns the same frame count.
        def _next_ltx_len(n):
            if n <= 1:
                return 1
            m = n - 1
            m = ((m + 7) // 8) * 8  # round up to multiple of 8
            return m + 1

        target_len = _next_ltx_len(n_real)
        pad = target_len - n_real
        if pad > 0:
            last = clip[-1]
            for _ in range(pad):
                clip.append(last.clone())
                # Pad entries are non-present sentinels; paste-back skips them.
                entries.append({"frame": -1, "x0": 0, "y0": 0, "win": 0,
                                "present": False, "run": -1})

        face_clip = torch.stack(clip, dim=0)  # [target_len, out_side, out_side, C]
        target_size = max(8, int(round(out_side * upscale_ratio)))
        data = {"entries": entries, "orig_shape": (B, H, W, C), "out_side": out_side,
                "upscale_ratio": float(upscale_ratio), "n_real": n_real,
                "n_runs": n_runs, "ltx_length": target_len}
        if n_runs > 1:
            print(f"[FaceTrackCropAndGate] {n_real} enhanced frames in {n_runs} "
                  f"separate runs (face crossed the threshold multiple times). "
                  f"They are concatenated into ONE clip of length {target_len} "
                  f"(padded by {pad}). A single LTX pass will bridge the run "
                  f"boundary; for strict per-run temporal coherence, process each "
                  f"run as its own LTX pass. Paste-back maps every frame correctly "
                  f"regardless.")
        elif pad > 0:
            print(f"[FaceTrackCropAndGate] padded clip {n_real}->{target_len} "
                  f"frames to a valid LTX length (8n+1); pad frames are ignored on "
                  f"paste-back.")
        return (face_clip, data, target_size, n_real, n_runs)


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

    RETURN_TYPES = ("IMAGE", "FACE_TRACK_DATA", "INT", "INT")
    RETURN_NAMES = ("run_clip", "run_track_data", "target_size", "enhanced_frames")
    FUNCTION = "select"
    CATEGORY = "masking/face_gate"
    DESCRIPTION = ("Extract one run from a multi-run face clip for an independent "
                   "LTX pass. Pad each run to its own valid LTX length.")

    @staticmethod
    def _next_ltx_len(n):
        if n <= 1:
            return 1
        return ((n - 1 + 7) // 8) * 8 + 1

    def select(self, face_clip, track_data, run_index):
        entries = track_data.get("entries", [])
        out_side = track_data.get("out_side", face_clip.shape[1] if face_clip.shape[0] else 8)
        ratio = track_data.get("upscale_ratio", 2.0)
        C = face_clip.shape[-1] if face_clip.ndim == 4 else 3

        # Indices (into face_clip / entries) of REAL frames in the requested run.
        sel = [k for k, e in enumerate(entries)
               if e.get("present", False) and e.get("run", -1) == run_index]
        if not sel:
            # Out-of-range / empty run. Return a 1-frame DUMMY clip (not 0 frames):
            # a 0-frame batch crashes downstream Resize/LTX (torch.stack on empty),
            # which would make over-provisioned branches fail. The single dummy
            # frame survives Resize->LTX->Decode harmlessly, and its lone entry is
            # present=False so paste-back skips it -> a true no-op branch. This is
            # what makes it safe to wire MORE run branches than a clip actually has.
            side = out_side if out_side and out_side > 0 else 8
            dummy = torch.zeros(1, side, side, C)
            data = {"entries": [{"frame": -1, "x0": 0, "y0": 0, "win": 0,
                                 "present": False, "run": -1}],
                    "orig_shape": track_data.get("orig_shape"),
                    "out_side": side, "upscale_ratio": float(ratio),
                    "n_real": 0, "n_runs": 0, "ltx_length": 1}
            return (dummy, data, max(8, int(round(side * ratio))), 0)

        run_frames = [face_clip[k] for k in sel]
        run_entries = [dict(entries[k]) for k in sel]  # copy; keep frame/x0/y0/win
        n_real = len(run_frames)

        # Pad THIS run to its own valid LTX length.
        target_len = self._next_ltx_len(n_real)
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
                "n_real": n_real, "n_runs": 1, "ltx_length": target_len}
        target_size = max(8, int(round(out_side * ratio)))
        return (run_clip, data, target_size, n_real)


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
                "feather": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01}),
                "only_present_frames": ("BOOLEAN", {"default": True,
                                        "tooltip": "Only composite frames where the face was actually detected."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "paste"
    CATEGORY = "masking/face_gate"
    DESCRIPTION = ("Resize the processed face clip back to its native window "
                   "(undoing upscale_ratio) and composite per frame.")

    def paste(self, original_images, processed_clip, track_data, feather, only_present_frames=True):
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
        expected = track_data.get("ltx_length", len(entries))
        if M != len(entries):
            raise ValueError(
                f"FaceTrackPasteBack: processed_clip has {M} frames but track_data "
                f"describes {len(entries)} (expected {expected}). They must match "
                f"1:1 — processed_clip[k] is the enhanced entries[k]. Most likely "
                f"the LTX 'length' is not wired from this clip's frame count "
                f"(use GetImageSizeAndCount on the upscaled face_clip -> "
                f"LTXVImgToVideo.length), or a node between the crop and here "
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


NODE_CLASS_MAPPINGS = {
    "FaceSizeGateMask": FaceSizeGateMask,
    "FaceCropAndGate": FaceCropAndGate,
    "FacePasteBack": FacePasteBack,
    "FaceTrackCropAndGate": FaceTrackCropAndGate,
    "FaceTrackSelectRun": FaceTrackSelectRun,
    "FaceTrackRunCount": FaceTrackRunCount,
    "FaceTrackPasteBack": FaceTrackPasteBack,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceSizeGateMask": "Face Size Gate (Mask)",
    "FaceCropAndGate": "Face Crop & Gate (per-face)",
    "FacePasteBack": "Face Paste Back (per-face)",
    "FaceTrackCropAndGate": "Face Track Crop & Gate (coherent)",
    "FaceTrackSelectRun": "Face Track Select Run (per-run)",
    "FaceTrackRunCount": "Face Track Run Count",
    "FaceTrackPasteBack": "Face Track Paste Back (coherent)",
}
