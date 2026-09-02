(function () {
  var KEY = "trtyolo-lang";

  function isEn(lang) {
    return lang === "en";
  }

  function apply(lang) {
    var en = isEn(lang);
    document.documentElement.lang = en ? "en" : "zh-CN";
    try {
      localStorage.setItem(KEY, en ? "en" : "zh");
    } catch (e) {}

    var title = document.querySelector("title");
    if (title) {
      var nextTitle = title.getAttribute(en ? "data-en" : "data-zh");
      if (nextTitle) document.title = nextTitle;
    }

    var meta = document.querySelector('meta[name="description"]');
    if (meta) {
      var nextDesc = meta.getAttribute(en ? "data-en" : "data-zh");
      if (nextDesc) meta.setAttribute("content", nextDesc);
    }

    document.querySelectorAll("[data-zh-alt]").forEach(function (el) {
      el.setAttribute("alt", el.getAttribute(en ? "data-en-alt" : "data-zh-alt"));
    });

    document.querySelectorAll("[data-set-lang]").forEach(function (btn) {
      btn.setAttribute("aria-pressed", btn.getAttribute("data-set-lang") === (en ? "en" : "zh") ? "true" : "false");
    });
  }

  document.addEventListener("click", function (e) {
    var btn = e.target.closest("[data-set-lang]");
    if (!btn) return;
    apply(btn.getAttribute("data-set-lang"));
  });

  apply(document.documentElement.lang === "en" ? "en" : "zh");
})();
