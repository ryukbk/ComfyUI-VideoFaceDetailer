import { app } from "../../scripts/app.js";

// Show only the size knob that is actually in use on Face Track Crop & Gate:
//   fixed_target_size OFF -> upscale_ratio   (target_size follows the face)
//   fixed_target_size ON  -> target_size     (one fixed edge length)
//
// Purely cosmetic. The Python backend reads whichever parameter the toggle selects
// (see _resolve_target_size), so the node behaves correctly even if this script never
// loads — both widgets simply stay visible.
//
// The two widgets are declared LAST in INPUT_TYPES, not next to each other, because
// ComfyUI serializes widget values positionally and inserting mid-list would shift
// every later value in already-saved workflows. Hiding the inactive one is what makes
// that on-disk ordering invisible on screen.

const MODE = "fixed_target_size";
const ON_WIDGET = "target_size";      // shown when the toggle is ON
const OFF_WIDGET = "upscale_ratio";   // shown when the toggle is OFF

function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

// Collapse a widget so it takes no space and is not drawn. `hidden` is the frontend's
// own extension hook (it drops the widget out of layout while leaving it in
// node.widgets, so the value still serializes); the type/computeSize pair is the older
// mechanism, kept so this works on either.
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
  const mode = findWidget(node, MODE);
  if (!mode) return;
  const fixed = !!mode.value;
  const onW = findWidget(node, ON_WIDGET);
  const offW = findWidget(node, OFF_WIDGET);
  if (fixed) {
    show(onW);
    hide(offW);
  } else {
    hide(onW);
    show(offW);
  }
  // Re-fit to the new widget set; never shrink the width the user chose.
  const sz = node.computeSize();
  node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "VideoFaceDetailer.SizeModeToggle",

  async nodeCreated(node) {
    if (node.comfyClass !== "FaceTrackCropAndGate") return;
    const mode = findWidget(node, MODE);
    if (!mode) return;
    const prev = mode.callback;
    mode.callback = function (...args) {
      const r = prev?.apply(this, args);
      applyVisibility(node);
      return r;
    };
    // Deserialising a saved workflow assigns widget values WITHOUT calling their
    // callbacks, so the initial state has to be applied separately - and deferred,
    // because nodeCreated runs before the widget values have been restored.
    setTimeout(() => applyVisibility(node), 0);
    const onConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      const r = onConfigure?.apply(this, args);
      setTimeout(() => applyVisibility(node), 0);
      return r;
    };
  },
});
