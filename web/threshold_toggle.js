import { app } from "../../scripts/app.js";

// Show only the threshold parameter that matches `threshold_type` on the
// Face Track Crop & Gate node, so the UI reflects which one is active:
//   width  -> max_width_fraction
//   height -> max_height_fraction
//   area   -> max_area_percent
// This is purely cosmetic — the Python backend always reads the parameter that
// matches threshold_type, so the node works correctly even if this script is
// absent (all three widgets simply stay visible).

const ACTIVE = {
  width: "max_width_fraction",
  height: "max_height_fraction",
  area: "max_area_percent",
};
const ALL = ["max_width_fraction", "max_height_fraction", "max_area_percent"];

function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

// Collapse a widget so it takes no space and is not drawn.
function hide(w) {
  if (!w || w.hidden) return;
  w._origType = w._origType ?? w.type;
  w._origCompute = w._origCompute ?? w.computeSize;
  w.type = "hidden";
  w.hidden = true;
  w.computeSize = () => [0, -4];
}
function show(w) {
  if (!w || !w.hidden) return;
  w.type = w._origType ?? "number";
  w.hidden = false;
  if (w._origCompute) w.computeSize = w._origCompute;
  else delete w.computeSize;
}

function applyVisibility(node) {
  const tw = findWidget(node, "threshold_type");
  if (!tw) return;
  const active = ACTIVE[tw.value] || "max_width_fraction";
  for (const name of ALL) {
    const w = findWidget(node, name);
    if (!w) continue;
    if (name === active) show(w);
    else hide(w);
  }
  // Re-fit the node to the new widget set.
  const sz = node.computeSize();
  node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "VideoFaceDetailer.ThresholdToggle",
  async nodeCreated(node) {
    if (node.comfyClass !== "FaceTrackCropAndGate") return;
    const tw = findWidget(node, "threshold_type");
    if (!tw) return;
    const prev = tw.callback;
    tw.callback = function (...args) {
      const r = prev?.apply(this, args);
      applyVisibility(node);
      return r;
    };
    // Defer initial apply until widgets are fully populated.
    setTimeout(() => applyVisibility(node), 0);
  },
});
