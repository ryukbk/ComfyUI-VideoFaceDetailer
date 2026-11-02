"""Measure every node of the example workflows as ComfyUI draws them.

    python3 measure_node_sizes.py [ComfyUI URL]

Both renderers draw a node at least as tall as its widgets need, which is often
more than the 320x200 build_ui_workflow.py starts from, and Nodes 2.0 (the Vue
node renderer) usually more than the classic canvas. This loads each workflow
in workflows/ in a running ComfyUI (default http://127.0.0.1:8188) that has the
workflows' node packs installed, once per renderer, in Chrome through
Playwright, and writes each node's larger drawn body size (below the title bar)
to node_sizes.json. build_ui_workflow.py lays the grid out from those sizes.
Run it again after changing a workflow's nodes, then rebuild.

Needs `pip install playwright` and Google Chrome.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "node_sizes.json"

MEASURE = """async ([wf, vue]) => {
    const previous = app.extensionManager.setting.get('Comfy.VueNodes.Enabled');
    await app.extensionManager.setting.set('Comfy.VueNodes.Enabled', vue);
    await app.loadGraphData(wf);
    await new Promise((resolve) => setTimeout(resolve, 3000));
    const ds = app.canvas.ds, canvas = app.canvas.canvas.getBoundingClientRect();
    const sizes = {};
    for (const node of app.graph._nodes) {
        let size = [node.size[0], node.size[1]];
        const needed = node.computeSize?.();
        if (needed) size = [Math.max(size[0], needed[0]), Math.max(size[1], needed[1])];
        if (vue) {
            const el = document.querySelector(`[data-node-id="${node.id}"]`);
            if (!el) continue;
            const box = el.getBoundingClientRect();
            // Body only: from pos (the title bar is drawn above it) to the drawn bottom.
            size = [box.width / ds.scale, (box.bottom - canvas.top) / ds.scale - ds.offset[1] - node.pos[1]];
        }
        sizes[node.id] = size;
    }
    await app.extensionManager.setting.set('Comfy.VueNodes.Enabled', previous);
    return sizes;
}"""


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8188"
    from playwright.sync_api import sync_playwright
    result = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1800, "height": 1100})
        page.goto(url)
        page.wait_for_function("() => window.app && app.graph && window.LiteGraph", timeout=120000)
        page.wait_for_timeout(1500)
        for path in sorted((ROOT / "workflows").glob("*_UI.json")):
            workflow = json.loads(path.read_text(encoding="utf-8"))
            missing = page.evaluate("(types) => types.filter((t) => !LiteGraph.registered_node_types[t])",
                                    sorted({n["type"] for n in workflow["nodes"]}))
            if missing:
                raise SystemExit("%s: install the node packs for %s first" % (path.name, ", ".join(missing)))
            sizes = {}
            for vue in (False, True):
                for node_id, (w, h) in page.evaluate(MEASURE, [workflow, vue]).items():
                    old = sizes.get(node_id, (0, 0))
                    sizes[node_id] = (max(old[0], w), max(old[1], h))
            # Rounded, not rounded up: Nodes 2.0 draws a node a fraction wider
            # than its saved size, so rounding up would grow it on every run.
            result[path.name] = {k: [round(w), round(h)]
                                 for k, (w, h) in sorted(sizes.items(), key=lambda kv: int(kv[0]))}
            print("measured %d nodes in %s" % (len(sizes), path.name))
        browser.close()
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print("-> %s" % OUT.name)


if __name__ == "__main__":
    main()
