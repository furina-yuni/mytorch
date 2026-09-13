# ruff: noqa: E501
"""Validate the generated multi-page MyTorch HTML API reference."""

from __future__ import annotations

import ast
import inspect
import json
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import mytorch as mt


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.assets: list[str] = []
        self.inline_styles = 0
        self.inline_scripts = 0
        self.api_cards = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if identifier := attributes.get("id"):
            self.ids.add(identifier)
        if tag == "a" and (href := attributes.get("href")):
            self.links.append(href)
        if tag == "link" and (href := attributes.get("href")):
            self.assets.append(href)
        if tag == "script":
            if source := attributes.get("src"):
                self.assets.append(source)
            else:
                self.inline_scripts += 1
        if tag == "style":
            self.inline_styles += 1
        classes = (attributes.get("class") or "").split()
        if "api-card" in classes:
            self.api_cards += 1


def _defined_functions(module: object) -> set[str]:
    exported = set(getattr(module, "__all__", ()))
    return {
        name
        for name, value in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_")
        and (value.__module__ == module.__name__ or name in exported)
    }


def _resolve_local(source: Path, reference: str) -> tuple[Path, str]:
    parts = urlsplit(reference)
    if parts.scheme or parts.netloc:
        return source, ""
    raw_path = unquote(parts.path)
    target = (source.parent / raw_path).resolve() if raw_path else source.resolve()
    return target, unquote(parts.fragment)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = root / "docs"
    catalog = json.loads(
        (docs / "content" / "modules.json").read_text(encoding="utf-8")
    )
    guide_catalog = json.loads(
        (docs / "content" / "guides.json").read_text(encoding="utf-8")
    )
    parameters = json.loads(
        (docs / "content" / "parameters.json").read_text(encoding="utf-8")
    )
    expected_pages = {docs / "index.html"}
    expected_pages |= {
        docs / "api" / f"{page['slug']}.html" for page in catalog["pages"]
    }
    expected_pages |= {
        docs / "guides" / f"{guide['slug']}.html" for guide in guide_catalog["guides"]
    }
    actual_pages = set(docs.rglob("*.html"))
    if actual_pages != expected_pages:
        missing = sorted(
            str(path.relative_to(docs)) for path in expected_pages - actual_pages
        )
        extra = sorted(
            str(path.relative_to(docs)) for path in actual_pages - expected_pages
        )
        raise AssertionError(
            f"documentation page mismatch: missing={missing}, extra={extra}"
        )

    documents: dict[Path, str] = {}
    parsers: dict[Path, _PageParser] = {}
    for page in sorted(actual_pages):
        document = page.read_text(encoding="utf-8")
        parser = _PageParser()
        parser.feed(document)
        documents[page.resolve()] = document
        parsers[page.resolve()] = parser
        if parser.inline_styles or parser.inline_scripts:
            raise AssertionError(
                f"inline CSS/JavaScript found in {page.relative_to(docs)}"
            )
        if mt.__version__ not in document:
            raise AssertionError(
                f"{page.relative_to(docs)} does not mention version {mt.__version__}"
            )

    for page, parser in parsers.items():
        for reference in parser.links + parser.assets:
            target, fragment = _resolve_local(page, reference)
            if target == page and not fragment and urlsplit(reference).scheme:
                continue
            if not target.exists():
                raise AssertionError(
                    f"broken local reference in {page.name}: {reference}"
                )
            if fragment:
                target_parser = parsers.get(target)
                if target_parser is None:
                    raise AssertionError(
                        f"fragment points outside HTML pages in {page.name}: {reference}"
                    )
                if fragment not in target_parser.ids:
                    raise AssertionError(f"missing anchor in {page.name}: {reference}")

    combined = "\n".join(documents.values())
    public_names = (
        set(mt.__all__)
        | set(mt.data.__all__)
        | set(mt.nn.__all__)
        | set(mt.optim.__all__)
    )
    public_names |= _defined_functions(mt.nn.functional)
    public_names |= _defined_functions(mt.nn.init)
    public_names |= {
        name
        for name, value in inspect.getmembers(mt.Tensor)
        if not name.startswith("_") and callable(value)
    }
    missing_names = sorted(name for name in public_names if name not in combined)
    if missing_names:
        raise AssertionError(f"undocumented public API names: {missing_names}")

    api_cards = sum(parser.api_cards for parser in parsers.values())
    if api_cards < len(public_names):
        raise AssertionError(
            f"only {api_cards} detailed API cards for {len(public_names)} public names"
        )
    if "이 API의 " in combined:
        raise AssertionError("generic fallback parameter descriptions remain")
    if len(parameters) < 100:
        raise AssertionError("parameter description catalog is unexpectedly small")

    css = (docs / "assets" / "css" / "docs.css").read_text(encoding="utf-8")
    javascript = (docs / "assets" / "js" / "docs.js").read_text(encoding="utf-8")
    signature_rules = ("white-space: pre-wrap", "overflow-wrap: anywhere")
    if not all(rule in css for rule in signature_rules):
        raise AssertionError(
            "API signatures are not configured to wrap within the page"
        )
    sidebar_rules = ("sidebarScrollKey", "sessionStorage", "pagehide")
    if not all(rule in javascript for rule in sidebar_rules):
        raise AssertionError("sidebar scroll persistence is not configured")
    guide_page = docs / "guides" / "training-and-inference.html"
    guide_document = documents[guide_page.resolve()]
    for guide in guide_catalog["guides"]:
        for code_key in ("training_code_file", "inference_code_file"):
            code_path = root / guide[code_key]
            code = code_path.read_text(encoding="utf-8")
            ast.parse(code, filename=str(code_path))
            if escape(code, quote=True) not in guide_document:
                raise AssertionError(f"guide does not embed {guide[code_key]}")
    guide_requirements = (
        "훈련 전체 코드",
        "추론 전체 코드",
        "mt.data.ImageFolder(",
        "mt.data.DataLoader(",
        "optimizer.zero_grad()",
        "with mt.amp.autocast():",
        "scaler.scale(loss).backward()",
        "scaler.step(optimizer)",
        "scheduler.step()",
        "model.eval()",
        "with mt.no_grad(), mt.amp.autocast():",
        "mt.save_checkpoint(",
        "mt.load_checkpoint(",
        "model.load_state_dict(",
    )
    if not all(value in guide_document for value in guide_requirements):
        raise AssertionError("training/inference guide is missing an essential step")

    print(f"HTML pages: {len(actual_pages)}")
    print(f"Detailed API cards: {api_cards}")
    print(f"Documented public API names: {len(public_names)}")
    print(f"Parameter descriptions: {len(parameters)}")
    print(f"MyTorch version: {mt.__version__}")
    print("Local links and external asset separation: OK")
    print("Sidebar persistence and signature wrapping: OK")
    print("End-to-end training and inference guide: OK")


if __name__ == "__main__":
    main()
