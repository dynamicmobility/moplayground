# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

This is the `docs/` subdirectory of the `moplayground` Python project — a static website that serves two purposes:

1. **Academic project page** (`index.html`) — a Bulma-styled landing page (adapted from the Nerfies / Academic Project Page Template) presenting MO-Playground: "Massively Parallelized Multi-Objective Reinforcement Learning for Robotics" by Neil Janwani, Ellen Novoseller, Vernon Lawhern, and Maegan Tucker.
2. **Jekyll documentation site** — Markdown pages (`installation.md`, `simulating.md`, `training.md`, `howto.md`, etc.) rendered with the `just-the-docs` remote theme, configured in `_config.yml`.

Both live side-by-side in the same directory: `index.html` declares `layout: null` and `nav_exclude: true` so Jekyll passes it through unchanged, while the `.md` files use the just-the-docs theme.

## Preview commands

Quick static preview (no Jekyll, won't render Markdown via theme):
```bash
python3 -m http.server
```

Full Jekyll build (renders `.md` pages with the just-the-docs theme — required to verify the docs site):
```bash
bundle install   # first time only
bundle exec jekyll serve
```

The `Gemfile` pins `jekyll ~> 4.3`, `just-the-docs`, and `jekyll-remote-theme`.

## Editing notes

- `index.html` is the landing page. Hero video timing is computed via inline JS keyed on `window.innerWidth` — adjust per-breakpoint start times in the `getStartTime()` function rather than re-encoding video.
- The just-the-docs theme is loaded as a `remote_theme` (no local copy). Theme customizations belong in `_config.yml` or via Jekyll's standard override directories.
- `howto.html`, `abstract.html`, `bruce_videos.html`, `env_videos.html` are extra HTML fragments; `howto.md` mirrors `howto.html` content for the docs theme.
- Static assets (videos, images, css) live under `static/`. The hero references `static/videos/bruce_swing_arms_banner.mp4`; the favicon is at `static/images/favicon.ico`.
- Content here documents the parent project at `../` — when commands or APIs in the docs (e.g. `python3 -m scripts.train`, `moplayground` package, `ral` package, conda envs from `environment.yml` / `mac_environment.yml`) drift from the actual code, check the parent repo before changing the docs.
- `docs/api/` is **auto-generated** from `moplayground` docstrings by `scripts/build_api_docs.py` (uses `lazydocs`). Don't hand-edit files in there — edit the docstrings in `src/moplayground/`, rerun the script, and commit the regenerated Markdown.
