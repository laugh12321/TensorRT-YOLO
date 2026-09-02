(function () {
  var COPIED_MS = 1600;
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

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
    if (!pre || pre.querySelector(".copy-btn") || pre.closest(".hero-panel") || pre.closest(".api-panel")) return;
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn copy-btn-pre";
    btn.setAttribute("data-copy", "");
    btn.setAttribute("aria-label", document.documentElement.lang === "en" ? "Copy" : "复制");
    btn.appendChild(iconUse("copy"));
    btn.appendChild(iconUse("check"));
    pre.appendChild(btn);
  });

  var scrolled = false;
  function onScroll() {
    var next = window.scrollY > 8;
    if (next === scrolled) return;
    scrolled = next;
    document.documentElement.classList.toggle("is-scrolled", scrolled);
  }
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  var reveals = document.querySelectorAll("[data-reveal]");
  function revealNow(el) {
    el.classList.add("is-in");
  }

  function isInView(el) {
    var rect = el.getBoundingClientRect();
    return rect.bottom > 0 && rect.top < (window.innerHeight || document.documentElement.clientHeight);
  }

  if (reduce || !("IntersectionObserver" in window)) {
    reveals.forEach(revealNow);
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        revealNow(entry.target);
        io.unobserve(entry.target);
      });
    }, { threshold: 0, rootMargin: "0px 0px -8% 0px" });
    reveals.forEach(function (el) {
      if (isInView(el)) revealNow(el);
      else io.observe(el);
    });
    document.documentElement.classList.add("js-reveal");
  }

  function decimalsOf(value) {
    var parts = String(value).split(".");
    return parts[1] ? parts[1].length : 0;
  }

  function formatCount(n, decimals, prefix, suffix) {
    return prefix + n.toFixed(decimals) + suffix;
  }

  function animateCount(el) {
    var target = parseFloat(el.getAttribute("data-count"));
    if (isNaN(target)) return;
    var prefix = el.getAttribute("data-prefix") || "";
    var suffix = el.getAttribute("data-suffix") || "";
    var decimals = decimalsOf(el.getAttribute("data-count"));
    if (reduce) {
      el.textContent = formatCount(target, decimals, prefix, suffix);
      return;
    }
    var duration = 900;
    var start = performance.now();
    function frame(now) {
      var t = Math.min(1, (now - start) / duration);
      var eased = 1 - Math.pow(1 - t, 3);
      el.textContent = formatCount(target * eased, decimals, prefix, suffix);
      if (t < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  var counters = document.querySelectorAll("[data-count]");
  if (!reduce) {
    counters.forEach(function (el) {
      var prefix = el.getAttribute("data-prefix") || "";
      var suffix = el.getAttribute("data-suffix") || "";
      el.textContent = formatCount(0, decimalsOf(el.getAttribute("data-count")), prefix, suffix);
    });
  }
  if (reduce || !("IntersectionObserver" in window)) {
    counters.forEach(animateCount);
  } else {
    var cio = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        animateCount(entry.target);
        cio.unobserve(entry.target);
      });
    }, { threshold: 0.4 });
    counters.forEach(function (el) { cio.observe(el); });
  }
})();
