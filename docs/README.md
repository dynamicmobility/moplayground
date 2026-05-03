# MO-Playground Website

This directory contains the MO-Playground project page (`index.html`) and the Jekyll-based documentation site (Markdown pages rendered with the `just-the-docs` remote theme).

## Quick preview (no Jekyll)

For a fast look at `index.html` and static assets only — Markdown docs will **not** be rendered with the theme:

```bash
python3 -m http.server
```

Then open http://localhost:8000.

## Full local build with Jekyll

Use this to preview the Markdown docs (`installation.md`, `training.md`, etc.) with the `just-the-docs` theme.

### Linux (Ubuntu/Debian)

```bash
# Install Ruby + build tools
sudo apt update
sudo apt install -y ruby-full build-essential zlib1g-dev

# Avoid installing gems as root — install to your user dir
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Install bundler and project dependencies
gem install bundler jekyll
cd docs
bundle install

# Serve
bundle exec jekyll serve
```

### macOS

```bash
# Install Ruby via Homebrew (system Ruby is too old / requires sudo)
brew install ruby

# Add Homebrew Ruby to PATH (use the path printed by `brew info ruby`)
echo 'export PATH="/opt/homebrew/opt/ruby/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Install bundler and project dependencies
gem install bundler jekyll
cd docs
bundle install

# Serve
bundle exec jekyll serve
```

Then open http://localhost:4000. Pass `--livereload` to auto-reload on file changes.

## Regenerating the API reference

The pages under `docs/api/` are auto-generated from `moplayground` docstrings by [`lazydocs`](https://github.com/ml-tooling/lazydocs). To preview changes to docstrings locally, run from the repo root inside the `moplayground` conda env:

```bash
pip install -e ".[docs]"      # one time — installs lazydocs
python scripts/build_api_docs.py
```

Rerun the script and commit `docs/api/` whenever you change docstrings in `src/moplayground/`.

<!-- # Academic Project Page Template

> **Update (September 2025)**: This template has been modernized with better design, SEO, and mobile support. For the original version, see the [original-version branch](https://github.com/eliahuhorwitz/Academic-project-page-template/tree/original-version).

A clean, responsive template for academic project pages.


Example project pages built using this template are:
- https://horwitz.ai/probex
- https://vision.huji.ac.il/probegen
- https://horwitz.ai/mother
- https://horwitz.ai/spectral_detuning
- https://vision.huji.ac.il/ladeda
- https://vision.huji.ac.il/dsire
- https://horwitz.ai/podd
- https://dreamix-video-editing.github.io
- https://horwitz.ai/conffusion
- https://horwitz.ai/3d_ads/
- https://vision.huji.ac.il/ssrl_ad
- https://vision.huji.ac.il/deepsim



## Start using the template
To start using the template click on `Use this Template`.

The template uses html for controlling the content and css for controlling the style. 
To edit the websites contents edit the `index.html` file. It contains different HTML "building blocks", use whichever ones you need and comment out the rest.  

**IMPORTANT!** Make sure to replace the `favicon.ico` under `static/images/` with one of your own, otherwise your favicon is going to be a dreambooth image of me.

## What's New

- Modern, clean design with better mobile support
- Improved SEO with proper meta tags and structured data
- Performance improvements (lazy loading, optimized assets)
- More Works dropdown
- Copy button for BibTeX citations
- Better accessibility

## Components

- Teaser video
- Image carousel
- YouTube video embedding
- Video carousel
- PDF poster viewer
- BibTeX citation

## Customization

The HTML file has TODO comments showing what to replace:

- Paper title, authors, institution, conference
- Links (arXiv, GitHub, etc.)
- Abstract and descriptions  
- Videos, images, and PDFs
- Related works in the dropdown
- Meta tags for SEO and social sharing

### Meta Tags
The template includes meta tags for better search engine visibility and social media sharing. These appear in the `<head>` section and help with:
- Google Scholar indexing
- Social media previews (Twitter, Facebook, LinkedIn)
- Search engine optimization

Create a 1200x630px social preview image at `static/images/social_preview.png`.

## Tips

- Compress images with [TinyPNG](https://tinypng.com)
- Use YouTube for large videos (>10MB)  
- Replace the favicon in `static/images/`
- Works with GitHub Pages

## Acknowledgments
Parts of this project page were adopted from the [Nerfies](https://nerfies.github.io/) page.

## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>. -->
