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
        "inputs": [("track_data", "SAM3_TRACK_DATA"), ("images", "IMAGE"), ("fps", "FLOAT")],
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
                    ("target_size", "INT"), ("enhanced_frames", "INT"), ("num_runs", "INT")],
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
        "inputs": [("image", "IMAGE"), ("width", "INT"), ("height", "INT")],
        "outputs": [("IMAGE", "IMAGE"), ("width", "INT"), ("height", "INT"), ("mask", "MASK")],
    },
    "LTXVImgToVideo": {
        # length converted from widget -> input so it can be driven by frame count.
        "inputs": [("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                   ("vae", "VAE"), ("image", "IMAGE"), ("width", "INT"), ("height", "INT"),
                   ("length", "INT")],
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
    "LTXVConditioning": {
        # frame_rate converted from widget -> input so it can be driven by source_fps.
        "inputs": [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("frame_rate", "FLOAT")],
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
        # frame_rate converted from widget -> input so it can be driven by source_fps.
        "inputs": [("images", "IMAGE"), ("frame_rate", "FLOAT")],
        "outputs": [("Filenames", "VHS_FILENAMES")],
    },
}

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
    "9":  ("SAM3_TrackPreview", [0.5], {"track_data": ("4", 0), "images": ("1", 0), "fps": ("22", 0)}),
    "7":  ("FaceTrackCropAndGate", [2.0, 0.10, 0.02, 0.3, 0.4, 0.5, 0.4],
           {"images": ("1", 0), "mask_track": ("24", 0)}),
    "8":  ("ImageResizeKJv2", ["lanczos", "stretch", "0, 0, 0", "center", 32],
           {"image": ("7", 0), "width": ("7", 2), "height": ("7", 2)}),
    # Count the upscaled face-clip frames; drives LTX length so it matches exactly.
    "23": ("GetImageSizeAndCount", [], {"image": ("8", 0)}),
    "10": ("CheckpointLoaderSimple", ["ltxv-2b.safetensors"], {}),
    "11": ("CLIPTextEncode", ["a sharp, detailed, high quality close-up of a human face, consistent identity"], {"clip": ("10", 1)}),
    "12": ("CLIPTextEncode", ["blurry, low quality, distorted, deformed, flicker"], {"clip": ("10", 1)}),
    # length is now an input (driven by frame count); remaining widgets: batch_size, strength.
    "13": ("LTXVImgToVideo", [1, 0.4],
           {"positive": ("11", 0), "negative": ("12", 0), "vae": ("10", 2),
            "image": ("8", 0), "width": ("7", 2), "height": ("7", 2), "length": ("23", 3)}),
    # frame_rate is now an input (driven by source_fps); no frame_rate widget value remains.
    "14": ("LTXVConditioning", [], {"positive": ("13", 0), "negative": ("13", 1), "frame_rate": ("22", 0)}),
    "15": ("KSampler", [0, "fixed", 30, 3.0, "euler", "normal", 0.4],
           {"model": ("10", 0), "positive": ("14", 0), "negative": ("14", 1), "latent_image": ("13", 2)}),
    "16": ("VAEDecode", [], {"samples": ("15", 0), "vae": ("10", 2)}),
    "20": ("FaceTrackPasteBack", [0.15, True],
           {"original_images": ("1", 0), "processed_clip": ("16", 0), "track_data": ("7", 1)}),
    # frame_rate is now an input (driven by source_fps); remaining widgets: loop_count,
    # filename_prefix, format, pingpong, save_output.
    "21": ("VHS_VideoCombine", [0, "face_enhanced_track", "video/h264-mp4", False, True],
           {"images": ("20", 0), "frame_rate": ("22", 0)}),
}

LAYOUT_TRACK = {"1": (0, 0), "2": (0, 1), "3": (1, 1), "4": (2, 0), "5": (3, 0),
                "24": (3, 1), "22": (1, 0), "9": (3, 3), "7": (4, 0),
                "8": (5, 0), "23": (5, 1), "10": (4, 3), "11": (5, 3), "12": (5, 4),
                "13": (6, 0), "14": (7, 0), "15": (8, 0), "16": (9, 0), "20": (10, 0),
                "21": (11, 0)}

# ── Graph B: per-frame, per-face variant ──
NODES_PERFACE = {
    "1":  ("VHS_LoadVideo", ["input.mp4", 0, 0, 0, 0, 0, 1], {}),
    "2":  ("CheckpointLoaderSimple", ["sam3.1.safetensors"], {}),
    "3":  ("CLIPTextEncode", ["face"], {"clip": ("2", 1)}),
    "4":  ("SAM3_Detect", [0.5, 2, False],
           {"model": ("2", 0), "image": ("1", 0), "conditioning": ("3", 0)}),
    "5":  ("FaceCropAndGate", [0.10, "bbox_width", 0.3, 512, 8],
           {"images": ("1", 0), "masks": ("4", 0)}),
    "6":  ("ImageResizeKJv2", [1024, 1024, "lanczos", "stretch", "0, 0, 0", "center", 32],
           {"image": ("5", 0)}),
    # Count the upscaled face frames; drives LTX length so it matches the crop batch.
    "23": ("GetImageSizeAndCount", [], {"image": ("6", 0)}),
    "10": ("CheckpointLoaderSimple", ["ltxv-2b.safetensors"], {}),
    "11": ("CLIPTextEncode", ["a sharp, detailed, high quality close-up of a human face"], {"clip": ("10", 1)}),
    "12": ("CLIPTextEncode", ["blurry, low quality, distorted, deformed"], {"clip": ("10", 1)}),
    # width/height/length all inputs now: width/height from resize outputs, length from count.
    "13": ("LTXVImgToVideo", [1, 0.4],
           {"positive": ("11", 0), "negative": ("12", 0), "vae": ("10", 2), "image": ("6", 0),
            "width": ("6", 1), "height": ("6", 2), "length": ("23", 3)}),
    "14": ("LTXVConditioning", [25.0], {"positive": ("13", 0), "negative": ("13", 1)}),
    "15": ("KSampler", [0, "fixed", 30, 3.0, "euler", "normal", 0.4],
           {"model": ("10", 0), "positive": ("14", 0), "negative": ("14", 1), "latent_image": ("13", 2)}),
    "16": ("VAEDecode", [], {"samples": ("15", 0), "vae": ("10", 2)}),
    "20": ("FacePasteBack", [0.15],
           {"original_images": ("1", 0), "processed_faces": ("16", 0), "face_data": ("5", 1)}),
    "21": ("VHS_VideoCombine", [0, "face_enhanced", "video/h264-mp4", False, True],
           {"images": ("20", 0), "frame_rate": ("22", 0)}),
    "22": ("VHS_VideoInfo", [], {"video_info": ("1", 3)}),
}
NODES_PERFACE["14"] = ("LTXVConditioning", [],
                       {"positive": ("13", 0), "negative": ("13", 1), "frame_rate": ("22", 0)})
LAYOUT_PERFACE = {"1": (0, 0), "2": (0, 1), "3": (1, 1), "4": (2, 0), "5": (3, 0),
                  "22": (1, 0), "6": (4, 0), "23": (4, 1), "10": (3, 3), "11": (4, 3),
                  "12": (4, 4), "13": (5, 0), "14": (6, 0), "15": (7, 0), "16": (8, 0),
                  "20": (9, 0), "21": (10, 0)}


def build(node_dict, layout, out_path):
    def pos(key):
        cx, cy = layout.get(key, (0, 0))
        return [cx * 360, cy * 220]

    nodes, links, link_id = [], [], 0
    order = list(node_dict.keys())

    for key in order:
        ntype, wv, conn = node_dict[key]
        spec = SPEC[ntype]
        node_inputs = [{"name": n, "type": t, "link": None} for (n, t) in spec["inputs"]]
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

    graph = {
        "last_node_id": max(int(k) for k in order), "last_link_id": link_id,
        "nodes": nodes, "links": links, "groups": [], "config": {}, "extra": {},
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
    "7":  ("FaceTrackCropAndGate", [2.0, 0.10, 0.02, 0.3, 0.4, 0.5, 0.4],
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
        rsz: ("ImageResizeKJv2", ["lanczos", "stretch", "0, 0, 0", "center", 32],
              {"image": (sel, 0), "width": (sel, 2), "height": (sel, 2)}),
        cnt: ("GetImageSizeAndCount", [], {"image": (rsz, 0)}),
        i2v: ("LTXVImgToVideo", [1, 0.4],
              {"positive": ("11", 0), "negative": ("12", 0), "vae": ("10", 2),
               "image": (rsz, 0), "width": (sel, 2), "height": (sel, 2), "length": (cnt, 3)}),
        cond: ("LTXVConditioning", [], {"positive": (i2v, 0), "negative": (i2v, 1), "frame_rate": ("22", 0)}),
        ks: ("KSampler", [0, "fixed", 30, 3.0, "euler", "normal", 0.4],
             {"model": ("10", 0), "positive": (cond, 0), "negative": (cond, 1), "latent_image": (i2v, 2)}),
        dec: ("VAEDecode", [], {"samples": (ks, 0), "vae": ("10", 2)}),
        pst: ("FaceTrackPasteBack", [0.15, True],
              {"original_images": prev_paste_ref, "processed_clip": (dec, 0), "track_data": (sel, 1)}),
    }
    return frag, (pst, 0)

_b0, _p0 = _branch_numeric(30, 0, ("1", 0))
_b1, _p1 = _branch_numeric(40, 1, _p0)
NODES_PERRUN.update(_b0)
NODES_PERRUN.update(_b1)
NODES_PERRUN["99"] = ("VHS_VideoCombine", [0, "face_enhanced_perrun", "video/h264-mp4", False, True],
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

build(NODES_TRACK, LAYOUT_TRACK, "face_enhance_ltx_track_workflow_UI.json")
build(NODES_PERFACE, LAYOUT_PERFACE, "face_enhance_ltx_workflow_UI.json")
build(NODES_PERRUN, LAYOUT_PERRUN, "face_enhance_ltx_track_perrun_workflow_UI.json")
