(function () {
  "use strict";

  const root = document.documentElement;
  const menuButton = document.querySelector(".menu-button");
  const themeButton = document.querySelector(".theme-button");
  const search = document.querySelector("#site-search");
  const results = document.querySelector("#search-results");
  const backToTop = document.querySelector(".back-to-top");
  const index = Array.isArray(window.MYTORCH_SEARCH_INDEX)
    ? window.MYTORCH_SEARCH_INDEX
    : [];

  const inApiDirectory = window.location.pathname.replaceAll("\\", "/").includes("/api/");
  const resolveUrl = (url) => (inApiDirectory ? `../${url}` : url);

  const savedTheme = localStorage.getItem("mytorch-docs-theme");
  if (savedTheme === "light" || savedTheme === "dark") {
    root.dataset.theme = savedTheme;
  }

  themeButton?.addEventListener("click", () => {
    const next = root.dataset.theme === "light" ? "dark" : "light";
    root.dataset.theme = next;
    localStorage.setItem("mytorch-docs-theme", next);
  });

  menuButton?.addEventListener("click", () => {
    const open = document.body.classList.toggle("menu-open");
    menuButton.setAttribute("aria-expanded", String(open));
  });

  document.querySelectorAll(".sidebar a").forEach((link) => {
    link.addEventListener("click", () => {
      document.body.classList.remove("menu-open");
      menuButton?.setAttribute("aria-expanded", "false");
    });
  });

  const escapeHtml = (value) =>
    value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

  function renderResults(query) {
    if (!results) return;
    const terms = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    if (terms.length === 0) {
      results.hidden = true;
      results.innerHTML = "";
      return;
    }
    const matches = index
      .filter((item) => {
        const haystack = `${item.title} ${item.module} ${item.summary}`.toLocaleLowerCase();
        return terms.every((term) => haystack.includes(term));
      })
      .slice(0, 14);
    results.innerHTML = matches.length
      ? matches
          .map(
            (item) =>
              `<a href="${resolveUrl(item.url)}"><code>${escapeHtml(item.title)}</code>` +
              `<small>${escapeHtml(item.module)}</small><span class="result-kind">${escapeHtml(item.kind)}</span></a>`,
          )
          .join("")
      : '<div class="search-empty">일치하는 공개 API가 없습니다.</div>';
    results.hidden = false;
  }

  search?.addEventListener("input", (event) => renderResults(event.target.value));
  search?.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      search.value = "";
      renderResults("");
      search.blur();
    }
    if (event.key === "Enter") {
      const first = results?.querySelector("a");
      if (first) first.click();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== search) {
      event.preventDefault();
      search?.focus();
    }
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".search-box") && results) results.hidden = true;
  });

  document.querySelectorAll(".copy-link").forEach((button) => {
    button.addEventListener("click", async () => {
      const target = button.dataset.copy;
      const url = `${window.location.href.split("#")[0]}#${target}`;
      try {
        await navigator.clipboard.writeText(url);
        button.textContent = "복사됨";
        window.setTimeout(() => {
          button.textContent = "링크 복사";
        }, 1400);
      } catch {
        window.location.hash = target;
      }
    });
  });

  window.addEventListener(
    "scroll",
    () => backToTop?.classList.toggle("visible", window.scrollY > 700),
    { passive: true },
  );
  backToTop?.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
})();
