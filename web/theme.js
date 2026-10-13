// Puts the chosen theme on <html> as `data-theme`. A classic script in <head>,
// after the stylesheets, rather than part of app.js: a module is deferred, so
// the page would paint once in the default theme and then flash to the chosen
// one. The legal pages load it too, so they match the app.
//
// The choice is per device, so it lives in localStorage, not on the account:
// "auto" (the default) follows the device's light/dark preference and keeps
// following it; "light", "dark" and "epaper" are fixed. Whatever changes the
// stored choice fires `themepick` on the document, and every application
// fires `themechange` for whoever draws with colours read from the stylesheet.
{
  const prefersDark = matchMedia("(prefers-color-scheme: dark)");
  const applyTheme = () => {
    const chosen = localStorage.getItem("theme") || "auto";
    const auto = prefersDark.matches ? "dark" : "light";
    document.documentElement.dataset.theme = chosen === "auto" ? auto : chosen;
    const panel = getComputedStyle(document.documentElement).getPropertyValue("--panel").trim();
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta && panel) meta.content = panel;
    document.dispatchEvent(new Event("themechange"));
  };
  applyTheme();
  prefersDark.addEventListener("change", applyTheme);
  document.addEventListener("themepick", applyTheme);
  addEventListener("storage", (e) => {
    if (e.key === "theme") applyTheme();
  });
}
