"""
Generate a ComfyUI **UI workflow-format** JSON (nodes + links + positions) for
the temporally-coherent face-enhance graph, so it loads correctly via drag/drop
(the API/prompt format does not).

Each node is declared explicitly with:
  type, widgets_values (in the node's widget order, ONLY for params not fed by a
  link), and conn = {input_name: (src_node_key, src_output_slot)}.

Input/output slot lists are declared per node so link slot indices are exact.
"""
import json

# (input slot names in order, output (name,type) list in order, widget param
#  names in order). Inputs that we connect via links are pulled out of widgets.
SPEC = {
    "VHS_LoadVideo": {
        "inputs": [],  # all widgets here; only outputs used
        "outputs": [("IMAGE", "IMAGE"), ("frame_count", "INT"),
                    ("audio", "VHS_AUDIO"), ("video_info", "VHS_VIDEOINFO")],
    },
    "CheckpointLoaderSimple": {
        "inputs": [],
        "outputs": [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
    },
    "CLIPTextEncode": {
        "inputs": [("clip", "CLIP")],
        "outputs": [("CONDITIONING", "CONDITIONING")],
    },
    "SAM3_VideoTrack": {
        "inputs": [("images", "IMAGE"), ("model", "MODEL"), ("conditioning", "CONDITIONING")],
        "outputs": [("track_data", "SAM3_TRACK_DATA")],
    },
    "SAM3_TrackToMask": {
        "inputs": [("track_data", "SAM3_TRACK_DATA")],
        "outputs": [("masks", "MASK")],
    },
    "SAM3_TrackPreview": {
        # fps converted from widget -> input so it can be driven by source_fps.
        # widget order: opacity, fps. fps is converted to an input (source_fps).
        "inputs": [("track_data", "SAM3_TRACK_DATA"), ("images", "IMAGE"), ("fps", "FLOAT", "widget")],
        "outputs": [],  # output node: renders a video preview, no tensor output
    },
    "VHS_VideoInfo": {
        "inputs": [("video_info", "VHS_VIDEOINFO")],
        # Slot 0 is source_fps (FLOAT). Names mirror the node's RETURN_NAMES.
        "outputs": [("source_fps", "FLOAT"), ("source_frame_count", "INT"),
                    ("source_duration", "FLOAT"), ("source_width", "INT"),
                    ("source_height", "INT"), ("loaded_fps", "FLOAT"),
                    ("loaded_frame_count", "INT"), ("loaded_duration", "FLOAT"),
                    ("loaded_width", "INT"), ("loaded_height", "INT")],
    },
    "SAM3_Detect": {
        "inputs": [("model", "MODEL"), ("image", "IMAGE"), ("conditioning", "CONDITIONING")],
        "outputs": [("masks", "MASK"), ("bboxes", "BOUNDING_BOX")],
    },
    "FaceCropAndGate": {
        "inputs": [("images", "IMAGE"), ("masks", "MASK")],
        "outputs": [("face_crops", "IMAGE"), ("face_data", "FACE_CROP_DATA"), ("face_count", "INT")],
    },
    "FacePasteBack": {
        "inputs": [("original_images", "IMAGE"), ("processed_faces", "IMAGE"), ("face_data", "FACE_CROP_DATA")],
        "outputs": [("images", "IMAGE")],
    },
    "FloatConstant": {
        "inputs": [],
        "outputs": [("value", "FLOAT")],
    },
    "FaceTrackCropAndGate": {
        # upscale_ratio is a plain WIDGET (not a linked input) so you edit it
        # directly on the node and target_size updates immediately.
        "inputs": [("images", "IMAGE"), ("mask_track", "MASK")],
        "outputs": [("face_clip", "IMAGE"), ("track_data", "FACE_TRACK_DATA"),
                    ("target_size", "INT"), ("enhanced_frames", "INT"), ("num_runs", "INT"),
                    ("frame_count", "INT"), ("enhanced", "BOOLEAN"), ("report", "STRING")],
    },
    "FaceTrackSelectRun": {
        "inputs": [("face_clip", "IMAGE"), ("track_data", "FACE_TRACK_DATA")],
        "outputs": [("run_clip", "IMAGE"), ("run_track_data", "FACE_TRACK_DATA"),
                    ("target_size", "INT"), ("enhanced_frames", "INT")],
    },
    "FaceTrackRunCount": {
        "inputs": [("track_data", "FACE_TRACK_DATA")],
        "outputs": [("num_runs", "INT"), ("enhanced_frames", "INT")],
    },
    "ImageResizeKJv2": {
        # width/height are widgets converted to inputs -> mark them (widget property)
        # so widgets_values stays aligned. Widget order (KJNodes latest):
        # width, height, upscale_method, keep_proportion, pad_color, crop_position,
        # divisible_by, device.
        "inputs": [("image", "IMAGE"), ("width", "INT", "widget"), ("height", "INT", "widget")],
        "outputs": [("IMAGE", "IMAGE"), ("width", "INT"), ("height", "INT"), ("mask", "MASK")],
    },
    "LTXVImgToVideo": {
        # widget order: width, height, length, batch_size, strength. width/height/
        # length are converted to inputs -> mark them so widgets_values stays aligned
        # (otherwise 'strength' silently falls back to its default 1.0).
        "inputs": [("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                   ("vae", "VAE"), ("image", "IMAGE"),
                   ("width", "INT", "widget"), ("height", "INT", "widget"),
                   ("length", "INT", "widget")],
        "outputs": [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")],
    },
    "GetImageSizeAndCount": {
        "inputs": [("image", "IMAGE")],
        "outputs": [("image", "IMAGE"), ("width", "INT"), ("height", "INT"), ("count", "INT")],
    },
    "GetMaskSizeAndCount": {
        # Shows "count x W x H" on the node; pass-through so it can sit inline.
        "inputs": [("mask", "MASK")],
        "outputs": [("mask", "MASK"), ("width", "INT"), ("height", "INT"), ("count", "INT")],
    },
    "MaskHasFace": {
        "inputs": [("masks", "MASK")],
        "outputs": [("has_face", "BOOLEAN"), ("frames_with_face", "INT")],
    },
    "LazySwitchKJ": {
        # on_true / on_false are lazy: the unchosen branch is never executed.
        "inputs": [("switch", "BOOLEAN", "widget"), ("on_false", "*"), ("on_true", "*")],
        "outputs": [("*", "*")],
    },
    "LTXVConditioning": {
        # frame_rate converted from widget -> input so it can be driven by source_fps.
        "inputs": [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("frame_rate", "FLOAT", "widget")],
        "outputs": [("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
    },
    "KSampler": {
        "inputs": [("model", "MODEL"), ("positive", "CONDITIONING"),
                   ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
        "outputs": [("LATENT", "LATENT")],
    },
    "VAEDecode": {
        "inputs": [("samples", "LATENT"), ("vae", "VAE")],
        "outputs": [("IMAGE", "IMAGE")],
    },
    "FaceTrackPasteBack": {
        "inputs": [("original_images", "IMAGE"), ("processed_clip", "IMAGE"),
                   ("track_data", "FACE_TRACK_DATA")],
        "outputs": [("IMAGE", "IMAGE")],
    },
    "VHS_VideoCombine": {
        # VHS stores widgets_values as a DICT (keyed by widget name), not a list —
        # a list makes the node render red. frame_rate is a widget converted to an
        # input (marked "widget" so it's link-driven by source_fps and not red);
        # audio (optional) muxes the ORIGINAL track onto the saved video.
        "inputs": [("images", "IMAGE"), ("frame_rate", "FLOAT", "widget"), ("audio", "AUDIO")],
        "outputs": [("Filenames", "VHS_FILENAMES")],
    },
    # ── MiniMax H3 resampler stage (see NODES_H3) ──
    "UNETLoader": {
        "inputs": [],
        "outputs": [("MODEL", "MODEL")],
    },
    "LoraLoaderModelOnly": {
        "inputs": [("model", "MODEL")],
        "outputs": [("MODEL", "MODEL")],
    },
    "CLIPLoader": {
        "inputs": [],
        "outputs": [("CLIP", "CLIP")],
    },
    "VAELoader": {
        "inputs": [],
        "outputs": [("VAE", "VAE")],
    },
    "LoadImage": {
        "inputs": [],
        "outputs": [("IMAGE", "IMAGE"), ("MASK", "MASK")],
    },
    "LoadAudio": {
        "inputs": [],
        "outputs": [("AUDIO", "AUDIO")],
    },
    "MiniMaxH3ReferenceToVideo": {
        # width/height/length converted from widgets -> inputs so they can be
        # driven by the resized face clip (H3 needs W/H %32 and length on 17k+5).
        # ref_images/ref_audios are autogrow input slots (names mirror ComfyUI).
        "inputs": [("clip", "CLIP"), ("vae", "VAE"), ("audio_vae", "VAE"),
                   ("ref_images.ref_image_0", "IMAGE"), ("ref_audios.ref_audio_0", "AUDIO"),
                   # width/height/length are widgets converted to inputs -> mark them
                   # so ComfyUI links slot<->widget (and widgets_values stays aligned).
                   ("width", "INT", "widget"), ("height", "INT", "widget"),
                   ("length", "INT", "widget")],
        "outputs": [("positive", "CONDITIONING"), ("LATENT", "LATENT")],
    },
    "H3FaceRefine": {
        # one node: img2img inject + audio lipsync + per-frame denoise. audio_vae/
        # audio are optional inputs (appended after the required connection slots).
        "inputs": [("model", "MODEL"), ("av_latent", "LATENT"), ("images", "IMAGE"),
                   ("vae", "VAE"), ("track_data", "FACE_TRACK_DATA"),
                   ("audio_vae", "VAE"), ("audio", "AUDIO")],
        "outputs": [("model", "MODEL"), ("av_latent", "LATENT"), ("report", "STRING")],
    },
    "FaceTrackAudioSlice": {
        # widget order: source_fps, target_fps. source_fps converted to input
        # (driven by VHS_VideoInfo source_fps) -> mark it.
        "inputs": [("audio", "AUDIO"), ("track_data", "FACE_TRACK_DATA"),
                   ("source_fps", "FLOAT", "widget")],
        "outputs": [("audio", "AUDIO"), ("report", "STRING")],
    },
    "BasicScheduler": {
        "inputs": [("model", "MODEL")],
        "outputs": [("SIGMAS", "SIGMAS")],
    },
    "BasicGuider": {
        "inputs": [("model", "MODEL"), ("conditioning", "CONDITIONING")],
        "outputs": [("GUIDER", "GUIDER")],
    },
    "ModelAttentionBackend": {
        # model/patch: swaps the attention backend on a cloned model. Sits between
        # the loader/refine node and the sampler.
        "inputs": [("model", "MODEL")],
        "outputs": [("MODEL", "MODEL")],
    },
    "KSamplerSelect": {
        "inputs": [],
        "outputs": [("SAMPLER", "SAMPLER")],
    },
    "RandomNoise": {
        "inputs": [],
        "outputs": [("NOISE", "NOISE")],
    },
    "SamplerCustomAdvanced": {
        "inputs": [("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"),
                   ("sigmas", "SIGMAS"), ("latent_image", "LATENT")],
        "outputs": [("output", "LATENT"), ("denoised_output", "LATENT")],
    },
}

def vhs_combine(filename_prefix):
    """VHS_VideoCombine widgets as the DICT format VHS expects (a positional list
    renders the node red). frame_rate is a placeholder — it's driven by the
    source_fps link at runtime."""
    return {"frame_rate": 30, "loop_count": 0, "filename_prefix": filename_prefix,
            "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19,
            "save_metadata": True, "trim_to_audio": False, "pingpong": False,
            "save_output": True, "videopreview": ""}


# ── Graph A: temporally-coherent track variant ──
# Declarative node list. key -> (type, widgets_values, conn)
NODES_TRACK = {
    "1":  ("VHS_LoadVideo", ["input.mp4", 0, 0, 0, 0, 0, 1], {}),
    "2":  ("CheckpointLoaderSimple", ["sam3.1.safetensors"], {}),
    "3":  ("CLIPTextEncode", ["face"], {"clip": ("2", 1)}),
    "4":  ("SAM3_VideoTrack", [0.5, 4, 1], {"images": ("1", 0), "model": ("2", 0), "conditioning": ("3", 0)}),
    "5":  ("SAM3_TrackToMask", ["0"], {"track_data": ("4", 0)}),
    # Inline mask count/size preview: shows "count x W x H" on the node, and
    # passes the mask through to the gate. This is the mask that actually feeds
    # node 7, so it reflects exactly what the gate sees (unlike SAM3_TrackPreview,
    # which reflects the raw track_data from node 4).
    "24": ("GetMaskSizeAndCount", [], {"mask": ("5", 0)}),
    # Video Info -> source_fps drives all frame-rate fields below.
    "22": ("VHS_VideoInfo", [], {"video_info": ("1", 3)}),
    # fps is now an input (driven by source_fps); no fps widget value remains.
    "9":  ("SAM3_TrackPreview", [0.5, 24.0], {"track_data": ("4", 0), "images": ("1", 0), "fps": ("22", 0)}),
    "7":  ("FaceTrackCropAndGate", [2.0, "width", 10.0, 0.0, 0.0, 0.3, 1.0, 0.5, 0.4],
           {"images": ("1", 0), "mask_track": ("24", 0)}),
    "8":  ("ImageResizeKJv2", [512, 512, "lanczos", "stretch", "0, 0, 0", "center", 32, "cpu"],
           {"image": ("7", 0), "width": ("7", 2), "height": ("7", 2)}),
    # Count the upscaled face-clip frames; drives LTX length so it matches exactly.
    "23": ("GetImageSizeAndCount", [], {"image": ("8", 0)}),
    "10": ("CheckpointLoaderSimple", ["ltxv-2b.safetensors"], {}),
    "11": ("CLIPTextEncode", ["a sharp, detailed, high quality close-up of a human face, consistent identity"], {"clip": ("10", 1)}),
    "12": ("CLIPTextEncode", ["blurry, low quality, distorted, deformed, flicker"], {"clip": ("10", 1)}),
    # length is now an input (driven by frame count); remaining widgets: batch_size, strength.
    "13": ("LTXVImgToVideo", [768, 512, 97, 1, 0.4],
           {"positive": ("11", 0), "negative": ("12", 0), "vae": ("10", 2),
            "image": ("8", 0), "width": ("7", 2), "height": ("7", 2), "length": ("23", 3)}),
    # frame_rate is now an input (driven by source_fps); no frame_rate widget value remains.
    "14": ("LTXVConditioning", [25.0], {"positive": ("13", 0), "negative": ("13", 1), "frame_rate": ("22", 0)}),
    "15": ("KSampler", [0, "fixed", 30, 3.0, "euler", "normal", 0.4],
           {"model": ("10", 0), "positive": ("14", 0), "negative": ("14", 1), "latent_image": ("13", 2)}),
    "16": ("VAEDecode", [], {"samples": ("15", 0), "vae": ("10", 2)}),
    "20": ("FaceTrackPasteBack", [0.15, "mask", True],
           {"original_images": ("1", 0), "processed_clip": ("16", 0), "track_data": ("7", 1)}),
    # LazySwitchKJ: on_true = the detailer output (node 20), on_false = original
    # video (node 1). The switch is driven by FaceTrackCropAndGate's `enhanced`
    # BOOLEAN (True iff >=1 frame qualified). The gate itself runs eagerly (it
    # drives the switch), but because on_true is LAZY, when `enhanced` is False the
    # rest of the branch (upscale -> LTX -> paste, nodes 8,13,15,16,20) is NOT
    # executed — so the no-op dummy never reaches Resize, and the original video
    # passes through. This also covers the no-face case (0 frames -> enhanced=False).
    "26": ("LazySwitchKJ", [False], {"switch": ("7", 6), "on_false": ("1", 0), "on_true": ("20", 0)}),
    # frame_rate is now an input (driven by source_fps); remaining widgets: loop_count,
    # filename_prefix, format, pingpong, save_output.
    "21": ("VHS_VideoCombine", vhs_combine("face_enhanced_track"),
           {"images": ("26", 0), "frame_rate": ("22", 0)}),
}

LAYOUT_TRACK = {"1": (0, 0), "2": (0, 1), "3": (1, 1), "4": (2, 0), "5": (3, 0),
                "24": (3, 1), "22": (1, 0), "9": (3, 3), "7": (4, 0),
                "8": (5, 0), "23": (5, 1), "10": (4, 3), "11": (5, 3), "12": (5, 4),
                "13": (6, 0), "14": (7, 0), "15": (8, 0), "16": (9, 0), "20": (10, 0),
                "26": (11, 1), "21": (12, 0)}

# ── Graph B: per-frame, per-face variant ──
NODES_PERFACE = {
    "1":  ("VHS_LoadVideo", ["input.mp4", 0, 0, 0, 0, 0, 1], {}),
    "2":  ("CheckpointLoaderSimple", ["sam3.1.safetensors"], {}),
    "3":  ("CLIPTextEncode", ["face"], {"clip": ("2", 1)}),
    "4":  ("SAM3_Detect", [0.5, 2, False],
           {"model": ("2", 0), "image": ("1", 0), "conditioning": ("3", 0)}),
    "5":  ("FaceCropAndGate", [0.10, "bbox_width", 0.3, 512, 8],
           {"images": ("1", 0), "masks": ("4", 0)}),
    "6":  ("ImageResizeKJv2", [1024, 1024, "lanczos", "stretch", "0, 0, 0", "center", 32, "cpu"],
           {"image": ("5", 0)}),
    # Count the upscaled face frames; drives LTX length so it matches the crop batch.
    "23": ("GetImageSizeAndCount", [], {"image": ("6", 0)}),
    "10": ("CheckpointLoaderSimple", ["ltxv-2b.safetensors"], {}),
    "11": ("CLIPTextEncode", ["a sharp, detailed, high quality close-up of a human face"], {"clip": ("10", 1)}),
    "12": ("CLIPTextEncode", ["blurry, low quality, distorted, deformed"], {"clip": ("10", 1)}),
    # width/height/length all inputs now: width/height from resize outputs, length from count.
    "13": ("LTXVImgToVideo", [768, 512, 97, 1, 0.4],
           {"positive": ("11", 0), "negative": ("12", 0), "vae": ("10", 2), "image": ("6", 0),
            "width": ("6", 1), "height": ("6", 2), "length": ("23", 3)}),
    "14": ("LTXVConditioning", [25.0], {"positive": ("13", 0), "negative": ("13", 1)}),
    "15": ("KSampler", [0, "fixed", 30, 3.0, "euler", "normal", 0.4],
           {"model": ("10", 0), "positive": ("14", 0), "negative": ("14", 1), "latent_image": ("13", 2)}),
    "16": ("VAEDecode", [], {"samples": ("15", 0), "vae": ("10", 2)}),
    "20": ("FacePasteBack", [0.15],
           {"original_images": ("1", 0), "processed_faces": ("16", 0), "face_data": ("5", 1)}),
    "21": ("VHS_VideoCombine", vhs_combine("face_enhanced"),
           {"images": ("20", 0), "frame_rate": ("22", 0)}),
    "22": ("VHS_VideoInfo", [], {"video_info": ("1", 3)}),
}
NODES_PERFACE["14"] = ("LTXVConditioning", [25.0],
                       {"positive": ("13", 0), "negative": ("13", 1), "frame_rate": ("22", 0)})
LAYOUT_PERFACE = {"1": (0, 0), "2": (0, 1), "3": (1, 1), "4": (2, 0), "5": (3, 0),
                  "22": (1, 0), "6": (4, 0), "23": (4, 1), "10": (3, 3), "11": (4, 3),
                  "12": (4, 4), "13": (5, 0), "14": (6, 0), "15": (7, 0), "16": (8, 0),
                  "20": (9, 0), "21": (10, 0)}


def build(node_dict, layout, out_path, groups=None):
    def pos(key):
        cx, cy = layout.get(key, (0, 0))
        return [cx * 360, cy * 220]

    nodes, links, link_id = [], [], 0
    order = list(node_dict.keys())

    for key in order:
        ntype, wv, conn = node_dict[key]
        spec = SPEC[ntype]
        # An input entry may be (name, type) for a genuine connection, or
        # (name, type, "widget") for a widget that has been converted to an input.
        # Converted-widget inputs get a "widget": {"name": …} property — this is how
        # ComfyUI associates the slot with its widget so the LINK drives the value
        # AND widgets_values (which must list ALL widgets in order) stays aligned.
        node_inputs = []
        for entry in spec["inputs"]:
            n, t = entry[0], entry[1]
            slot = {"name": n, "type": t, "link": None}
            if len(entry) > 2 and entry[2] == "widget":
                slot["widget"] = {"name": n}
            node_inputs.append(slot)
        node_outputs = [{"name": n, "type": t, "links": [], "slot_index": i}
                        for i, (n, t) in enumerate(spec["outputs"])]
        nodes.append({
            "id": int(key), "type": ntype, "pos": pos(key), "size": [320, 200],
            "flags": {}, "order": order.index(key), "mode": 0,
            "inputs": node_inputs, "outputs": node_outputs,
            "properties": {"Node name for S&R": ntype}, "widgets_values": wv,
        })

    node_by_id = {n["id"]: n for n in nodes}
    for key in order:
        ntype, wv, conn = node_dict[key]
        spec = SPEC[ntype]
        in_names = [s[0] for s in spec["inputs"]]
        tgt_node = node_by_id[int(key)]
        for iname, (src_key, src_slot) in conn.items():
            if iname not in in_names:
                raise SystemExit(f"BUG: {ntype} has no input slot '{iname}'")
            tslot = in_names.index(iname)
            src_spec = SPEC[node_dict[src_key][0]]
            if src_slot >= len(src_spec["outputs"]):
                raise SystemExit(f"BUG: {node_dict[src_key][0]} has no output slot {src_slot}")
            ltype = src_spec["outputs"][src_slot][1]
            link_id += 1
            links.append([link_id, int(src_key), src_slot, int(key), tslot, ltype])
            tgt_node["inputs"][tslot]["link"] = link_id
            node_by_id[int(src_key)]["outputs"][src_slot]["links"].append(link_id)

    # Optional group boxes. Each group's bounding is computed to enclose all of its
    # member node keys (with header + margin), so selecting the group in ComfyUI
    # selects exactly those nodes — e.g. for "Convert to Subgraph".
    group_list = []
    for g in (groups or []):
        xs, ys = [], []
        for k in g["keys"]:
            x, y = pos(k)
            xs.append(x); ys.append(y)
        x0, y0 = min(xs) - 30, min(ys) - 70          # header space above
        x1, y1 = max(xs) + 340, max(ys) + 240        # node ~320x200 + margin
        group_list.append({
            "title": g["title"], "bounding": [x0, y0, x1 - x0, y1 - y0],
            "color": g.get("color", "#3f789e"), "font_size": 24, "flags": {},
        })

    graph = {
        "last_node_id": max(int(k) for k in order), "last_link_id": link_id,
        "nodes": nodes, "links": links, "groups": group_list, "config": {}, "extra": {},
        "version": 0.4,
    }
    with open(out_path, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"wrote {out_path}  (nodes: {len(nodes)}, links: {link_id})")


# ── Graph C: per-run-pass reference variant (handles up to 2 runs) ─────────────
# small->large->small splits enhanced frames into runs. Here each run gets its
# OWN LTX pass (no cross-run bridging), then paste-backs are chained. Extend to
# more runs by replicating the SelectRun->Resize->Count->LTX->Decode->PasteBack
# block with run_index = 2, 3, ... (out-of-range runs are no-ops, so extra
# branches are harmless on clips with fewer runs).
NODES_PERRUN = {
    "1":  ("VHS_LoadVideo", ["input.mp4", 0, 0, 0, 0, 0, 1], {}),
    "2":  ("CheckpointLoaderSimple", ["sam3.1.safetensors"], {}),
    "3":  ("CLIPTextEncode", ["face"], {"clip": ("2", 1)}),
    "4":  ("SAM3_VideoTrack", [0.5, 4, 1], {"images": ("1", 0), "model": ("2", 0), "conditioning": ("3", 0)}),
    "5":  ("SAM3_TrackToMask", ["0"], {"track_data": ("4", 0)}),
    "22": ("VHS_VideoInfo", [], {"video_info": ("1", 3)}),
    "7":  ("FaceTrackCropAndGate", [2.0, "width", 10.0, 0.0, 0.0, 0.3, 1.0, 0.5, 0.4],
           {"images": ("1", 0), "mask_track": ("5", 0)}),
    "10": ("CheckpointLoaderSimple", ["ltxv-2b.safetensors"], {}),
    "11": ("CLIPTextEncode", ["a sharp, detailed, high quality close-up of a human face, consistent identity"], {"clip": ("10", 1)}),
    "12": ("CLIPTextEncode", ["blurry, low quality, distorted, deformed, flicker"], {"clip": ("10", 1)}),
}
# Each run gets its own LTX pass + paste-back. Run 0 pastes onto the ORIGINAL
# video; run 1 pastes onto run 0's result (chained). Out-of-range runs are no-ops.
def _branch_numeric(base, run_index, prev_paste_ref):
    sel, rsz, cnt, i2v, cond, ks, dec, pst = (str(base+i) for i in range(8))
    frag = {
        sel: ("FaceTrackSelectRun", [run_index], {"face_clip": ("7", 0), "track_data": ("7", 1)}),
        rsz: ("ImageResizeKJv2", [512, 512, "lanczos", "stretch", "0, 0, 0", "center", 32, "cpu"],
              {"image": (sel, 0), "width": (sel, 2), "height": (sel, 2)}),
        cnt: ("GetImageSizeAndCount", [], {"image": (rsz, 0)}),
        i2v: ("LTXVImgToVideo", [768, 512, 97, 1, 0.4],
              {"positive": ("11", 0), "negative": ("12", 0), "vae": ("10", 2),
               "image": (rsz, 0), "width": (sel, 2), "height": (sel, 2), "length": (cnt, 3)}),
        cond: ("LTXVConditioning", [25.0], {"positive": (i2v, 0), "negative": (i2v, 1), "frame_rate": ("22", 0)}),
        ks: ("KSampler", [0, "fixed", 30, 3.0, "euler", "normal", 0.4],
             {"model": ("10", 0), "positive": (cond, 0), "negative": (cond, 1), "latent_image": (i2v, 2)}),
        dec: ("VAEDecode", [], {"samples": (ks, 0), "vae": ("10", 2)}),
        pst: ("FaceTrackPasteBack", [0.15, "mask", True],
              {"original_images": prev_paste_ref, "processed_clip": (dec, 0), "track_data": (sel, 1)}),
    }
    return frag, (pst, 0)

_b0, _p0 = _branch_numeric(30, 0, ("1", 0))
_b1, _p1 = _branch_numeric(40, 1, _p0)
NODES_PERRUN.update(_b0)
NODES_PERRUN.update(_b1)
NODES_PERRUN["99"] = ("VHS_VideoCombine", vhs_combine("face_enhanced_perrun"),
                      {"images": _p1, "frame_rate": ("22", 0)})
# num_runs is now output slot 4 of FaceTrackCropAndGate (node 7). After a run the
# crop node's console log also states it. To SEE the value on the canvas, wire
# (7, 4) into any INT display node (e.g. easy-use "Show Any"); not wired here to
# avoid a hard dependency.

LAYOUT_PERRUN = {"1": (0, 0), "2": (0, 1), "3": (1, 1), "4": (2, 0), "5": (3, 0),
                 "22": (1, 0), "7": (4, 0), "10": (4, 4), "11": (5, 4), "12": (5, 5)}
# place branch nodes in two rows
for j, base in enumerate((30, 40)):
    for i in range(8):
        LAYOUT_PERRUN[str(base + i)] = (5 + i, j)
LAYOUT_PERRUN["99"] = (13, 0)

# ── Graph D: MiniMax H3 ref2va resampler with lipsync ─────────────────────────
# Same tracked-face front-end as Graph A, but the LTX resample block is replaced
# by an H3 img2img stage: the crop node pads to H3's 17k+5 grid
# (resampler="minimax_h3"); MiniMaxH3ReferenceToVideo builds the joint AV latent
# with identity refs on ref_image_0; the single H3FaceRefine node then seeds the
# VIDEO stream with the real upscaled face clip (frame-faithful img2img), locks the
# ORIGINAL audio into the AUDIO stream with video-only denoise (lip-sync), and
# scales denoise by face size — all in one node; SamplerCustomAdvanced + VAEDecode
# return the refined frames; FaceTrackPasteBack composites them (colour_match on);
# the ORIGINAL audio is muxed at save. Modeled on ComfyUI-H3-FaceRefine
# (Carasibana)'s proven wiring.
NODES_H3 = {
    "1":  ("VHS_LoadVideo", ["input.mp4", 0, 0, 0, 0, 0, 1], {}),
    "2":  ("CheckpointLoaderSimple", ["sam3.1.safetensors"], {}),
    "3":  ("CLIPTextEncode", ["face"], {"clip": ("2", 1)}),
    "4":  ("SAM3_VideoTrack", [0.5, 4, 1], {"images": ("1", 0), "model": ("2", 0), "conditioning": ("3", 0)}),
    "5":  ("SAM3_TrackToMask", ["0"], {"track_data": ("4", 0)}),
    "24": ("GetMaskSizeAndCount", [], {"mask": ("5", 0)}),
    "22": ("VHS_VideoInfo", [], {"video_info": ("1", 3)}),
    "9":  ("SAM3_TrackPreview", [0.5, 24.0], {"track_data": ("4", 0), "images": ("1", 0), "fps": ("22", 0)}),
    # resampler="minimax_h3" -> clip padded to H3's 17k+5 grid.
    "7":  ("FaceTrackCropAndGate",
           [2.0, "width", 10.0, 0.0, 0.0, 0.3, 1.0, 0.5, 0.4, "minimax_h3"],
           {"images": ("1", 0), "mask_track": ("24", 0)}),
    # ×32 canvas (H3 needs width/height divisible by 32).
    "8":  ("ImageResizeKJv2", [512, 512, "lanczos", "stretch", "0, 0, 0", "center", 32, "cpu"],
           {"image": ("7", 0), "width": ("7", 2), "height": ("7", 2)}),
    # ── H3 models ──
    "40": ("UNETLoader", ["minimax_h3_ref2va.safetensors", "default"], {}),
    "41": ("LoraLoaderModelOnly", ["minimax_h3_fl2v_lightx2v_turbo_4step.safetensors", 0.75],
           {"model": ("40", 0)}),
    "42": ("CLIPLoader", ["qwen3vl_32b_minimax_h3.safetensors", "minimax"], {}),
    "43": ("VAELoader", ["minimax_h3_video_vae_fp16.safetensors"], {}),
    "44": ("VAELoader", ["minimax_h3_audio_vae_fp32.safetensors"], {}),
    "45": ("LoadImage", ["identity_reference.png"], {}),
    # ref2va latent: identity ref + the ORIGINAL audio as a reference; W/H/length
    # driven by the clip. ref_image_size="max" uses the ref at up to 2048px instead
    # of downscaling it to the (small) face canvas — without this the reference has
    # almost no effect at a small canvas (the "ref image is ignored" bug).
    # Prompt follows MiniMax H3's reference-conditioned STRUCTURED format
    # (VIDEO_PROMPT_WRITING_GUIDE_ref_en.md): the six ordered sections
    # subject_definitions / summary / retention_analysis / detailed_description /
    # overall_soundscape / non_diegetic_music. It MUST cite each reference with its
    # tag or the model ignores it: <Picture 1> = ref_images.ref_image_0 (identity),
    # <Audio 1> = ref_audios.ref_audio_0 (the sliced source speech for lip-sync).
    # retention_analysis uses fully_preserved for the identity/frame and fully_copy
    # for the audio (we lock the original audio exactly).
    # widget order is [prompt, width, height, length, ref_image_size]; width/height/
    # length are also inputs (the links drive them) but their widget values MUST be
    # present so ref_image_size="max" lands on the right widget, not "match".
    "47": ("MiniMaxH3ReferenceToVideo",
           ["subject_definitions:\n"
            "<Subject 1> is the person in <Picture 1>. <Audio 1> is the speech/voice "
            "reference for <Subject 1> (S1) — the spoken vocal track the mouth must follow.\n\n"
            "summary:\n"
            "[reference generation + audio reference] The target video is a sharp, "
            "detailed close-up of <Subject 1> speaking naturally, keeping the exact "
            "identity from <Picture 1> and lip-syncing precisely to <Audio 1>.\n\n"
            "retention_analysis:\n"
            "<Subject 1> (appears in [Shot 1]): fully_preserved - the exact facial "
            "features, identity and skin texture from <Picture 1> are retained.\n"
            "<Picture 1> ([Shot 1] first frame): fully_preserved - used as the identity "
            "and appearance anchor for the face.\n"
            "<Audio 1>: fully_copy - the mouth shapes and speech timing follow this "
            "audio exactly for lip-sync.\n\n"
            "detailed_description:\n"
            "The target video is a high-quality, sharp, naturally-lit close-up in a "
            "realistic photographic style.\n"
            "[Shot 1] <Subject 1> (S1), the person from <Picture 1>, faces the camera in "
            "soft natural light, preserving the exact identity, facial features and skin "
            "texture from <Picture 1>. Speaking naturally and lip-syncing precisely to "
            "<Audio 1>, the lips, mouth and jaw move in time with the speech while the "
            "identity stays consistent.\n\n"
            "overall_soundscape:\n"
            "The spoken voice from <Audio 1> is the only diegetic sound; keep it clean "
            "with no added ambience.\n\n"
            "non_diegetic_music:\n"
            "None.", 1344, 768, 124, "max"],
           {"clip": ("42", 0), "vae": ("43", 0), "audio_vae": ("44", 0),
            "ref_images.ref_image_0": ("45", 0), "ref_audios.ref_audio_0": ("57", 0),
            "width": ("8", 1), "height": ("8", 2), "length": ("7", 5)}),
    # Audio sliced to the gated clip's frames (so H3 lip-sync matches the enhanced
    # frames, not the whole timeline). Full original audio is muxed at save.
    "57": ("FaceTrackAudioSlice", [24.0, 24.0],
           {"audio": ("1", 2), "track_data": ("7", 1), "source_fps": ("22", 0)}),
    # One node: img2img inject (face clip) + lip-sync (original audio) + per-frame
    # denoise. Outputs the patched model and the prepared AV latent.
    "48": ("H3FaceRefine", [1.0, 0.35, 1.0, 9],
           {"model": ("56", 0), "av_latent": ("47", 1), "images": ("8", 0), "vae": ("43", 0),
            "track_data": ("7", 1), "audio_vae": ("44", 0), "audio": ("57", 0)}),
    # Attention backend override, applied right after the Lora loader (falls back
    # to PyTorch attention if the "comfy kitchen attention" kernels aren't installed).
    "56": ("ModelAttentionBackend", ["comfy kitchen attention"], {"model": ("41", 0)}),
    # H3FaceRefine outputs the patched model (attention + audio lock) for the sampler.
    "51": ("BasicGuider", [], {"model": ("48", 0), "conditioning": ("47", 0)}),
    "52": ("BasicScheduler", ["simple", 4, 0.45], {"model": ("48", 0)}),
    "53": ("KSamplerSelect", ["res_multistep"], {}),
    "54": ("RandomNoise", [42, "fixed"], {}),
    "55": ("SamplerCustomAdvanced", [],
           {"noise": ("54", 0), "guider": ("51", 0), "sampler": ("53", 0),
            "sigmas": ("52", 0), "latent_image": ("48", 1)}),
    "16": ("VAEDecode", [], {"samples": ("55", 0), "vae": ("43", 0)}),
    # colour_match=1.0 -> match refined face tone to the original region (edge seam).
    "20": ("FaceTrackPasteBack", [0.15, "mask", True, 1.0],
           {"original_images": ("1", 0), "processed_clip": ("16", 0), "track_data": ("7", 1)}),
    # LazySwitchKJ: on_true = detailer output (20), on_false = original video (1).
    # The switch is driven by FaceTrackCropAndGate's `enhanced` BOOLEAN (True iff
    # >=1 frame qualified). The gate runs eagerly to drive the switch; on_true is
    # LAZY, so when `enhanced` is False the whole H3 chain (8,47,48,55,16,20) is
    # NOT executed — no wasted H3 pass, no model load, and the no-op dummy never
    # reaches Resize (which would collapse to 0 under divisible_by=32). Covers both
    # the no-face case and the "faces present but none qualify" case in one signal.
    "26": ("LazySwitchKJ", [False], {"switch": ("7", 6), "on_false": ("1", 0), "on_true": ("20", 0)}),
    # original audio muxed onto the saved video (H3 assumes 24fps for lipsync).
    "21": ("VHS_VideoCombine", vhs_combine("face_enhanced_h3"),
           {"images": ("26", 0), "frame_rate": ("22", 0), "audio": ("1", 2)}),
}
# Layout is split into two bands so the subgraph nodes form one clean rectangle:
#   Row 0  = nodes that stay OUTSIDE the subgraph (loaders, LoadVideo, VideoInfo,
#            ModelAttentionBackend, VideoCombine).
#   Rows 2+ = nodes that go INSIDE the subgraph (SAM3 -> gate -> resize -> H3 ->
#            decode -> paste -> switch). A group box (GROUPS_H3) encloses exactly
#            these, so "Convert to Subgraph" yields the intended 8-in/1-out boundary.
LAYOUT_H3 = {
    # ── OUTSIDE (top band, row 0) ──
    "1": (0, 0), "22": (1, 0), "45": (2, 0), "40": (3, 0), "41": (4, 0),
    "56": (5, 0), "42": (6, 0), "43": (7, 0), "44": (8, 0), "21": (9, 0),
    # ── INSIDE the subgraph (rows 2+) ──
    "2": (0, 2), "3": (1, 2), "4": (2, 2), "5": (3, 2), "24": (4, 2), "9": (5, 2),
    "7": (0, 3), "8": (1, 3), "57": (2, 3),
    "47": (0, 4), "48": (1, 4), "51": (3, 4), "52": (4, 4), "53": (5, 4), "54": (6, 4),
    "55": (0, 5),
    "16": (0, 6), "20": (1, 6), "26": (2, 6),
}
# Nodes that become the subgraph (everything except the outside band above).
H3_SUBGRAPH_KEYS = ["2", "3", "4", "5", "24", "9", "7", "8", "57",
                    "47", "48", "51", "52", "53", "54", "55", "16", "20", "26"]
GROUPS_H3 = [{
    "title": "H3 Face Detailer — select this group, then 'Convert to Subgraph'",
    "keys": H3_SUBGRAPH_KEYS, "color": "#3f789e",
}]

build(NODES_TRACK, LAYOUT_TRACK, "workflows/face_enhance_ltx_track_workflow_UI.json")
build(NODES_PERFACE, LAYOUT_PERFACE, "workflows/face_enhance_ltx_workflow_UI.json")
build(NODES_PERRUN, LAYOUT_PERRUN, "workflows/face_enhance_ltx_track_perrun_workflow_UI.json")
build(NODES_H3, LAYOUT_H3, "workflows/face_enhance_h3_track_workflow_UI.json", groups=GROUPS_H3)
