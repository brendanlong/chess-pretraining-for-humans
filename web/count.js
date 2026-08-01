// The page counter: one request per page load, built here rather than by
// GoatCounter's own count.js.
//
// Their script derives what it reports from the page, which is the one part we
// can't delegate: it sends the full query string (as `p`, and again raw as `q`,
// which no dashboard setting suppresses) plus document.title. A share link puts
// an item id in the query string, so their script would ship a piece of the
// research record — which item a person was shown — to a third party, and no
// configuration on their side can stop it.
//
// So the reported path is a constant looked up in the table below, never
// anything read off the URL. A page that isn't in the table reports *nothing*
// until someone adds it deliberately, which makes counting a new page a
// decision rather than a default. That is also why this is worth the fifty
// lines: a sanitizer is one unanticipated URL away from leaking, and a closed
// vocabulary isn't.
//
// The upside beyond privacy is that no third-party script means `script-src`
// stays 'self' with no SRI pin or upstream version to track — the counter is
// now only an origin in `connect-src` and `img-src` (see trainer/server.py,
// which a test holds against the endpoint below).

// Must match server.ANALYTICS_BEACON, which allows it through the CSP.
const ENDPOINT = "https://chess-pretraining.goatcounter.com/count";

// Every page that counts, and what it counts as. `/` and `/index.html` are one
// page reached two ways, so they report as one.
const COUNTED = {
  "/": "/",
  "/index.html": "/",
  "/terms.html": "/terms.html",
  "/privacy.html": "/privacy.html",
};

// GoatCounter filters bots server-side by user agent; this is the signal it
// can't see from there. 153 is its documented code for a driven browser.
const BOT_WEBDRIVER = 153;

// GoatCounter's own local-traffic filter is server-side and by IP, so a machine
// reached over a tailnet — a public-looking name pointing at a private address
// — counts as real traffic. That is how a laptop or a staging box pollutes the
// live dashboard, so the check is repeated here on the hostname.
function isLocal(hostname) {
  return (
    /^(localhost|\[?::1\]?|0\.0\.0\.0)$|\.localhost$|^127\.|^10\.|^192\.168\.|^172\.(1[6-9]|2\d|3[01])\./.test(
      hostname,
    ) || hostname.endsWith(".ts.net")
  );
}

// Where people arrive from is the one thing worth knowing that the page itself
// can't say — but document.referrer is the *previous* page's full URL, and when
// that page was ours it carries the share link's item id: the same leak the
// path table exists to prevent, arriving through a second door. So: reported
// only when it is cross-origin, and then only its origin.
function crossOriginReferrer() {
  try {
    const url = new URL(document.referrer);
    return url.origin === location.origin ? null : url.origin;
  } catch {
    return null; // no referrer, or one that won't parse
  }
}

function countUrl(path) {
  const params = new URLSearchParams({
    p: path,
    // Screen width only: GoatCounter buckets it, and height and pixel ratio are
    // fingerprinting surface for a number nobody reads.
    s: String(screen.width),
    // Some browsers ignore the endpoint's cache headers.
    rnd: Math.random().toString(36).slice(2, 7),
  });
  // `t` (the title) is never sent. Ours name the app rather than the trial
  // today, and a counter that is fed titles is one page title away from
  // carrying the record again. GoatCounter shows the path instead.
  const referrer = crossOriginReferrer();
  if (referrer) params.set("r", referrer);
  if (navigator.webdriver) params.set("b", String(BOT_WEBDRIVER));
  return `${ENDPOINT}?${params.toString()}`;
}

function count() {
  if (!Object.hasOwn(COUNTED, location.pathname) || isLocal(location.hostname)) return;
  const url = countUrl(COUNTED[location.pathname]);
  // sendBeacon survives the page unloading, but can be refused — by a CSP, an
  // extension, or a browser without it. Fall back to an image, and let a
  // counter that can't count be the quietest failure in the app.
  try {
    if (navigator.sendBeacon?.(url)) return;
  } catch {
    // refused; the image below is the fallback GoatCounter's own script uses
  }
  try {
    new Image().src = url;
  } catch {
    // nothing about the app depends on this working
  }
}

count();
