(function () {
  var COPIED_MS = 1600;

  function spriteHref(id) {
    var scripts = document.getElementsByTagName("script");
    var src = "";
    for (var i = 0; i < scripts.length; i++) {
      if (scripts[i].src && /site\.js(?:\?|$)/.test(scripts[i].src)) {
        src = scripts[i].src;
        break;
      }
    }
    return src.replace(/site\.js(?:\?.*)?$/, "icons.svg") + "#" + id;
  }

  function iconUse(id) {
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "icon icon-" + id);
    svg.setAttribute("aria-hidden", "true");
    var use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", spriteHref(id));
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.appendChild(use);
    return svg;
  }

  function markCopied(btn) {
    btn.classList.add("is-copied");
    window.setTimeout(function () {
      btn.classList.remove("is-copied");
    }, COPIED_MS);
  }

  function copyFrom(btn) {
    var sel = btn.getAttribute("data-copy");
    var target = sel ? document.querySelector(sel) : btn.closest("pre");
    if (!target) return;
    var text = target.tagName === "CODE" || target.matches("code") ? target.textContent : (target.querySelector("code") || target).textContent;
    if (!navigator.clipboard) return;
    navigator.clipboard.writeText(text).then(function () {
      markCopied(btn);
    });
  }

  document.addEventListener("click", function (e) {
    var btn = e.target.closest("[data-copy]");
    if (!btn) return;
    copyFrom(btn);
  });

  document.querySelectorAll("pre > code").forEach(function (code) {
    var pre = code.parentElement;
    if (!pre || pre.querySelector(".copy-btn") || pre.closest(".hero-panel")) return;
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn copy-btn-pre";
    btn.setAttribute("data-copy", "");
    btn.setAttribute("aria-label", document.documentElement.lang === "en" ? "Copy" : "复制");
    btn.appendChild(iconUse("copy"));
    btn.appendChild(iconUse("check"));
    pre.appendChild(btn);
  });
})();
