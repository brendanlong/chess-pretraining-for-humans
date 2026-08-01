// Builds web/ into web-dist/: bundle, minify, copy the rest.
//
// The build's only job is to make the files smaller and fewer. It does not
// rename anything, rewrite any reference, or decide anything about caching —
// trainer/assets.py does all of that at startup, reading whichever tree it is
// pointed at, so there is one implementation of it rather than one per tree.
// That is what keeps a dev checkout (no build, serves web/) honest about what
// the image serves: the difference is bytes, not behaviour.
//
// No `target` is set on purpose. esbuild then assumes esnext and only
// minifies; naming a lower target would let it *transpile*, which is a change
// in semantics to fix a compatibility problem this app doesn't have — the
// source is hand-written for browsers and eslint already pins the level.

import { cp, mkdir, readdir, rm, stat } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as esbuild from "esbuild";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "web");
const out = join(root, "web-dist");

// The files a page links directly. Each pulls in whatever it imports —
// app.js takes chessground with it, board.css takes the three board
// stylesheets — which is why vendor/ is not copied.
const ENTRIES = ["app.js", "count.js", "board.css", "style.css"];
// Inlined into the entries above; shipping them too would be dead weight
// nothing references.
const BUNDLED_AWAY = "vendor";

await rm(out, { recursive: true, force: true });
await mkdir(out, { recursive: true });

const result = await esbuild.build({
  entryPoints: ENTRIES.map((e) => join(src, e)),
  outdir: out,
  bundle: true,
  minify: true,
  format: "esm",
  logLevel: "warning",
  metafile: true,
});

const copied = [];
for (const entry of await readdir(src, { withFileTypes: true })) {
  if (entry.name === BUNDLED_AWAY || ENTRIES.includes(entry.name)) continue;
  await cp(join(src, entry.name), join(out, entry.name), { recursive: true });
  copied.push(entry.name);
}

// A .js or .css that is neither an entry nor reached from one would be copied
// through unminified and still work, so this can't be a silent difference in
// what ships — say it, and let whoever added the file decide.
const unbundled = copied.filter((n) => n.endsWith(".js") || n.endsWith(".css"));
if (unbundled.length) {
  console.warn(`warning: copied unbundled, add to ENTRIES if a page links it: ${unbundled}`);
}

const inputs = Object.keys(result.metafile.inputs).length;
let before = 0;
for (const path of Object.keys(result.metafile.inputs)) {
  before += (await stat(join(root, path))).size;
}
let after = 0;
for (const [, meta] of Object.entries(result.metafile.outputs)) after += meta.bytes;
const pct = Math.round((1 - after / before) * 100);
console.log(
  `web-dist: ${ENTRIES.length} bundles from ${inputs} sources, ` +
    `${before} -> ${after} bytes (-${pct}%); ${copied.length} files copied`,
);
