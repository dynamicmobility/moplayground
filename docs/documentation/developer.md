# Developer guide

## Tests

```bash
pip install -e ".[dev]"
pytest              # fast tier: CPU-only unit tests (~40 s), also run in CI
pytest -m gpu       # GPU tier: trains tiny MORLAX, AMOR and PPO runs end to end (~4 min)
pytest tests/unit/test_configs_dir.py --config-dir path/to/configs   # check another config directory
```

GitHub Actions runs the fast tier on every push to `main` and every pull request
(`.github/workflows/tests.yml`). Known bugs are pinned as `xfail(strict=True)`;
fixing one makes its test pass unexpectedly and fail, so remove the marker in the same
change.

## Building these docs

The site is built with [MkDocs](https://www.mkdocs.org/) and the
[Material](https://squidfunk.github.io/mkdocs-material/) theme. The API reference is
generated from docstrings by [mkdocstrings](https://mkdocstrings.github.io/).

```bash
pip install -e ".[docs]"
mkdocs serve            # live preview at http://127.0.0.1:8000/moplayground/
mkdocs build --strict   # what CI runs; fails on broken links and references
```

The configuration is `mkdocs.yml` at the repository root. `docs/` holds both the
project landing page and these pages:

```text
docs/
├── index.html            # project landing page (copied through unchanged)
├── abstract.html, ...    # sections loaded by the landing page
├── static/               # landing page CSS, images, videos
└── documentation/        # this site
    ├── index.md          # home page
    ├── examples/         # walkthroughs of scripts/train.py and scripts/rollout.py
    ├── guides/           # the config file, environments, algorithms
    ├── api/              # one page per subpackage, generated from docstrings
    └── developer.md      # this page
```

`.github/workflows/docs.yml` builds the site on every pull request and publishes it to
GitHub Pages on every push to `main`.

### Writing docstrings

The API pages pick up every public function, class and method automatically.
Underscored names are left out. Docstrings use the
[Google style](https://mkdocstrings.github.io/griffe/reference/docstrings/#google-style):

```python
def load_mo_policy(algorithm, network_params, num_objectives, run_dir, tradeoff):
    """Load the latest multi-objective policy in ``run_dir``.

    Args:
        algorithm: ``'morlax'`` or ``'amor'``.
        tradeoff: Objective weights to condition the policy on.

    Returns:
        A function ``(obs, rng) -> (action, extras)``.
    """
```

To link to an API entry from any page, use its full path as a Markdown reference:
`` [`load_mo_policy`][moplayground.learning.inference.load_mo_policy] ``.

A new module needs a `::: moplayground.<subpackage>.<module>` line on its subpackage's
page in `docs/documentation/api/`.

### Keeping the examples in sync

The example pages quote excerpts of `scripts/train.py` and `scripts/rollout.py`. The
full listing at the bottom of each page is included straight from the file, so it
never goes stale, but the excerpts and their explanations must be updated by hand when
the scripts change.
