# GitHub Copilot Instructions

## Project Overview

This is a personal portfolio and blog website for André Fernandes, a Portuguese software engineer based in Porto, Portugal. The site is built with **Jekyll** and hosted on **GitHub Pages**. It showcases professional experience, academic background, and technical blog posts covering topics such as machine learning, performance optimization, distributed systems, and big data.

## Tech Stack

- **Static Site Generator:** Jekyll (Ruby)
- **Templating:** Liquid (`{{ }}` for output, `{% %}` for logic)
- **Styling:** SCSS — Bootstrap 4 partials (`assets/css/bootstrap/`) plus custom styles in `assets/css/main.scss`
- **Frontend Libraries:** jQuery 3.6.0, Popper.js, Bootstrap 4, Bootstrap Icons (`bi-*`)
- **Content:** Markdown with YAML front matter for pages and blog posts
- **Data:** YAML files under `_data/` for navbar, resume, and technologies
- **JavaScript Libraries:** Bundled locally under `assets/js/` (jQuery, Bootstrap, Popper.js, html2canvas)

## Directory Structure

```
_config.yml          # Jekyll site configuration
_data/               # YAML data files (navbar.yaml, resume.yml, technologies.yaml)
_includes/           # Reusable Liquid components (navbar, posts, social)
_layouts/            # Page templates (default.html, home.html, resume.html)
_posts/              # Blog posts (Markdown, named YYYY-MM-DD-slug.{md,markdown})
about/               # About page
academics/           # Academic background page
assets/
  css/               # main.scss + Bootstrap SCSS partials
  img/               # Images: profile, tech logos, post assets
  js/                # Bundled JS libraries
posts/               # Posts index page
prof-exp/            # Professional experience page
resume/              # Resume page
technologies/        # Technologies showcase page
```

## Coding Conventions

### Liquid Templates
- Layout files live in `_layouts/` and include partials from `_includes/`.
- Use Bootstrap grid classes (`col-`, `row`, `container`) for responsive layout.
- Use Bootstrap Icons classes (`bi-*`) for icons.
- Terminal-style UI elements use the CSS classes `.zsh-footer` and `.zsh-prompt`.

### SCSS / CSS
- All custom styles are added in `assets/css/main.scss`.
- Bootstrap SCSS partials are in `assets/css/bootstrap/`; do not modify them directly.
- Follow the existing Bootstrap 4 utility-class conventions.

### Blog Posts
- File name format: `YYYY-MM-DD-slug.md` or `YYYY-MM-DD-slug.markdown` inside `_posts/`.
- Every post must include YAML front matter:
  ```yaml
  ---
  layout: post
  title:  "Post Title"
  date:   YYYY-MM-DD HH:MM:SS +0000
  permalink: /posts/slug/
  categories: category1 category2
  image: /assets/img/posts/topic/image.png
  ---
  ```
- Posts use Markdown with fenced code blocks and language identifiers (e.g., ` ```python `).
- Post images are stored in `assets/img/posts/<topic>/`.

### Data Files (`_data/`)
- `navbar.yaml` — navigation links and metadata; add new sections here when adding pages.
- `resume.yml` — resume/CV content rendered by the resume layout.
- `technologies.yaml` — technology icons and labels shown on the technologies page.

### Jekyll Configuration
- Site-wide settings (title, author, plugins, theme) are in `_config.yml`.
- Active plugins: `jekyll-feed`, `jemoji`, `github-pages`.

## Development

To run the site locally:

```bash
bundle install
bundle exec jekyll serve --livereload
```

The site will be available at `http://localhost:4000`.
