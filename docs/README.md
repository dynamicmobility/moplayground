# MO-Playground website

This directory is the source of the site published at
https://dynamicmobility.github.io/moplayground/. It holds two parts, built together
by [MkDocs](https://www.mkdocs.org/) from `mkdocs.yml` at the repository root:

- **Landing page**: `index.html`, the HTML sections it loads (`abstract.html`,
  `env_videos.html`, ...) and `static/`. MkDocs copies these through unchanged.
- **Documentation**: the Markdown pages under `documentation/`, rendered with the
  Material theme. The API reference is generated from the `moplayground` docstrings by
  mkdocstrings.

Preview and build from the repository root:

```bash
pip install -e ".[docs]"
mkdocs serve            # http://127.0.0.1:8000/moplayground/
mkdocs build --strict   # what CI runs
```

`.github/workflows/docs.yml` publishes the site to the `gh-pages` branch on every push
to `main`. See `documentation/developer.md` for more.
