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


def add_frontmatter() -> list[str]:
    """Prepend just-the-docs frontmatter to every generated module page.

    Returns the list of module titles, sorted, for the index page.
    """
    titles: list[str] = []
    for md in sorted(API_DIR.glob("*.md")):
        if md.name == "index.md":
            continue
        title = md.stem  # lazydocs uses dotted module names as filenames
        titles.append(title)
        body = md.read_text()
        frontmatter = (
            "---\n"
            "layout: default\n"
            f'title: "{title}"\n'
            "parent: API Reference\n"
            "---\n\n"
        )
        md.write_text(frontmatter + body)
    return titles


def write_index(titles: list[str]) -> None:
    lines = [
        "---",
        "layout: default",
        "title: API Reference",
        "nav_order: 7",
        "has_children: true",
        "---",
        "",
        "# API Reference",
        "",
        "Auto-generated from the docstrings of the `moplayground` package.",
        "Regenerate with `python scripts/build_api_docs.py` after editing docstrings in `src/moplayground/`.",
        "",
        "## Modules",
        "",
    ]
    for title in titles:
        lines.append(f"- [{title}]({{% link api/{title}.md %}})")
    lines.append("")
    (API_DIR / "index.md").write_text("\n".join(lines))


def main() -> None:
    run_lazydocs()
    titles = add_frontmatter()
    write_index(titles)
    print(f"Wrote {len(titles)} module pages to {API_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
