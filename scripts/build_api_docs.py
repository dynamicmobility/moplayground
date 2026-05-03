"""Generate API reference Markdown for the docs site.

Runs `lazydocs` against the `moplayground` package, writes the output to
`docs/api/`, and prepends just-the-docs frontmatter to each generated file
so they slot into the Jekyll sidebar under the "API Reference" parent page.

Run from the repo root inside the moplayground conda env:

    python scripts/build_api_docs.py
"""
from __future__ import annotations

import shutil
from importlib.machinery import FileFinder
from pathlib import Path


def _patch_filefinder_for_py312() -> None:
    """Restore ``FileFinder.find_module`` removed in Python 3.12.

    ``lazydocs`` (>=0.4.8) calls ``loader.find_module(name).load_module(name)``
    where ``name`` is the *full dotted* module path. ``FileFinder.find_module``
    was removed in 3.12, and naively forwarding to ``find_spec`` returns a
    ``SourceFileLoader`` keyed on the leaf name, which then refuses to
    ``load_module`` the dotted name. Sidestep both by returning a tiny shim
    whose ``load_module`` defers to ``importlib.import_module``. Safe no-op
    on <3.12.
    """
    if hasattr(FileFinder, "find_module"):
        return

    import importlib

    class _ImportlibLoader:
        def load_module(self, fullname):  # noqa: ANN001
            return importlib.import_module(fullname)

    def find_module(self, fullname, path=None):  # noqa: ANN001
        leaf = fullname.rpartition(".")[2]
        if self.find_spec(leaf) is None:
            return None
        return _ImportlibLoader()

    FileFinder.find_module = find_module  # type: ignore[attr-defined]

REPO_ROOT = Path(__file__).resolve().parent.parent
API_DIR = REPO_ROOT / "docs" / "api"
PACKAGE = "moplayground"
SRC_BASE_URL = "https://github.com/dynamicmobility/moplayground/blob/main/"


def run_lazydocs() -> None:
    if API_DIR.exists():
        shutil.rmtree(API_DIR)
    API_DIR.mkdir(parents=True)

    _patch_filefinder_for_py312()
    from lazydocs import generate_docs

    generate_docs(
        paths=[PACKAGE],
        output_path=str(API_DIR),
        src_base_url=SRC_BASE_URL,
        watermark=False,
    )


def add_frontmatter() -> dict[str, list[str]]:
    """Prepend just-the-docs frontmatter to every generated module page.

    Pages are nested by top-level subpackage: ``moplayground.envs`` is a
    child of "API Reference" with ``has_children: true``, and any deeper
    module (``moplayground.envs.create``, ``moplayground.envs.generic``,
    ``moplayground.envs.generic.mobase``) becomes a child of
    ``moplayground.envs`` with ``grand_parent: API Reference``. Just-the-docs
    only supports 3 nav levels, so subpackages deeper than two segments
    are flattened under their top-level subpackage.

    Returns a mapping ``subpackage -> [child titles]`` for use by the index.
    """
    files = sorted(p for p in API_DIR.glob("*.md") if p.name != "index.md")
    titles = [f.stem for f in files]
    subpackages = sorted({t for t in titles if t.count(".") == 1})
    children: dict[str, list[str]] = {sp: [] for sp in subpackages}

    for md in files:
        title = md.stem
        body = md.read_text()
        if title.count(".") == 1:
            frontmatter = (
                "---\n"
                "layout: default\n"
                f'title: "{title}"\n'
                "parent: API Reference\n"
                "has_children: true\n"
                "---\n\n"
            )
        else:
            subpackage = ".".join(title.split(".")[:2])
            children.setdefault(subpackage, []).append(title)
            frontmatter = (
                "---\n"
                "layout: default\n"
                f'title: "{title}"\n'
                f'parent: "{subpackage}"\n'
                "grand_parent: API Reference\n"
                "---\n\n"
            )
        md.write_text(frontmatter + body)

    for sp in children:
        children[sp].sort()
    return children


def write_index(children: dict[str, list[str]]) -> None:
    lines = [
        "---",
        "layout: default",
        "title: API Reference",
        "nav_order: 7",
        "has_children: true",
        "has_toc: false",
        "---",
        "",
        "# API Reference",
        "",
        "Auto-generated from the docstrings of the `moplayground` package.",
        "Regenerate with `python scripts/build_api_docs.py` after editing docstrings in `src/moplayground/`.",
        "",
    ]
    for subpackage in sorted(children):
        lines.append(f"## [{subpackage}]({{% link api/{subpackage}.md %}})")
        lines.append("")
        for child in children[subpackage]:
            lines.append(f"- [{child}]({{% link api/{child}.md %}})")
        lines.append("")
    (API_DIR / "index.md").write_text("\n".join(lines))


def main() -> None:
    run_lazydocs()
    children = add_frontmatter()
    write_index(children)
    total = len(children) + sum(len(v) for v in children.values())
    print(f"Wrote {total} module pages to {API_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
