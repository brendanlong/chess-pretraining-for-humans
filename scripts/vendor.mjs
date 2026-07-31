// Copies chessground's shipped files into web/vendor/ so the frontend can
// reach them at a real URL.
//
// The pages load natively — a browser resolves `./vendor/chessground.min.js`
// and `@import "vendor/chessground.base.css"`, but not a bare `chessground`
// specifier, and teaching it one would mean an import map that the bundler
// then has to be kept agreeing with. Copying instead means the source tree a
// developer runs is the same tree the build reads, and the only thing pinning
// the version is package.json.
//
// Runs from `postinstall`, so a fresh clone has a working web/ after `npm ci`
// and no separate step to forget. web/vendor/ is gitignored for the same
// reason it used to be committed by hand and shouldn't be: it is output.

import { copyFile, mkdir, readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const pkg = join(root, "node_modules", "chessground");
const dest = join(root, "web", "vendor");

// Kept explicit rather than globbed: these four are what the frontend
// references, and a chessground release that adds a fifth shouldn't silently
// start shipping it.
const FILES = [
  ["dist/chessground.min.js", "chessground.min.js"],
  ["assets/chessground.base.css", "chessground.base.css"],
  ["assets/chessground.brown.css", "chessground.brown.css"],
  ["assets/chessground.cburnett.css", "chessground.cburnett.css"],
];

const { version } = JSON.parse(await readFile(join(pkg, "package.json"), "utf8"));
await mkdir(dest, { recursive: true });
for (const [from, to] of FILES) {
  await copyFile(join(pkg, from), join(dest, to));
}
console.log(`vendored chessground ${version} -> web/vendor/`);
