# Documentation Development

This directory contains the source files for the FLARES-2 Gridder documentation, built with [MkDocs](https://www.mkdocs.org/) and the [Material theme](https://squidfunk.github.io/mkdocs-material/).

## Quick Start

### Install Dependencies

```bash
pip install -r requirements-docs.txt
```

### Build and Serve Locally

```bash
# Serve with live reload (recommended for development)
mkdocs serve

# Build static HTML
mkdocs build
```

The documentation will be available at `http://127.0.0.1:8000/`

### Deploy to GitHub Pages

```bash
# Manual deployment
mkdocs gh-deploy

# Automatic deployment via GitHub Actions (recommended)
git push origin main  # Workflow triggers automatically
```

## Documentation Structure

```
docs/
├── index.md                      # Homepage
├── getting-started/
│   ├── installation.md          # Installation guide
│   ├── quickstart.md            # Quick start tutorial
│   ├── configuration.md         # Environment configuration
│   └── parameters.md            # Parameter file reference
├── performance/
│   ├── openmp.md               # OpenMP threading guide
│   └── mpi.md                  # MPI parallelization guide
└── javascripts/
    └── mathjax.js              # MathJax configuration

mkdocs.yml                       # MkDocs configuration
requirements-docs.txt            # Python dependencies
```

## Writing Documentation

### Markdown Features

MkDocs supports extended Markdown via the Material theme:

#### Code Blocks with Syntax Highlighting

````markdown
```cpp
#pragma omp parallel for
for (size_t i = 0; i < n; i++) {
  // OpenMP parallel loop
}
```
````

#### Admonitions (Callout Boxes)

```markdown
!!! note
    This is a note admonition.

!!! warning
    This is a warning admonition.

!!! tip
    This is a tip admonition.
```

#### Tables

```markdown
| Parameter | Type | Description |
|-----------|------|-------------|
| nkernels | int | Number of kernels |
| cdim | int | Grid dimension |
```

#### Mathematical Equations

Inline: `\( E = mc^2 \)`

Block:
```markdown
\[
\delta(\mathbf{x}) = \frac{\rho(\mathbf{x}) - \bar{\rho}}{\bar{\rho}}
\]
```

### Navigation Structure

Edit `mkdocs.yml` to modify the navigation:

```yaml
nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting-started/installation.md
    - Quick Start: getting-started/quickstart.md
```

## Adding New Pages

1. **Create the Markdown file** in the appropriate directory:
   ```bash
   touch docs/user-guide/new-page.md
   ```

2. **Add to navigation** in `mkdocs.yml`:
   ```yaml
   nav:
     - User Guide:
       - New Page: user-guide/new-page.md
   ```

3. **Preview changes**:
   ```bash
   mkdocs serve
   ```

## Deployment

### Automatic (Recommended)

Documentation is automatically deployed to GitHub Pages when changes are pushed to the `main` branch:

1. Edit documentation files
2. Commit and push to `main`:
   ```bash
   git add docs/
   git commit -m "Update documentation"
   git push origin main
   ```
3. GitHub Actions builds and deploys automatically
4. View at: `https://USERNAME.github.io/REPOSITORY/`

### Manual

Deploy directly from your local machine:

```bash
mkdocs gh-deploy --force
```

This builds the site and pushes to the `gh-pages` branch.

## Configuration

### mkdocs.yml

Key configuration options:

```yaml
site_name: FLARES-2 Gridder
theme:
  name: material
  palette:
    - scheme: default      # Light mode
    - scheme: slate        # Dark mode
  features:
    - navigation.tabs      # Top-level tabs
    - navigation.sections  # Collapsible sections
    - search.suggest       # Search suggestions
    - content.code.copy    # Copy button on code blocks
```

### Theme Customization

To customize colors:

```yaml
theme:
  palette:
    primary: indigo
    accent: blue
```

Available colors: red, pink, purple, deep purple, indigo, blue, light blue, cyan, teal, green, light green, lime, yellow, amber, orange, deep orange

## GitHub Actions Workflow

The deployment workflow (`.github/workflows/deploy-docs.yml`) triggers on:

- Pushes to `main` branch affecting documentation files
- Manual workflow dispatch from GitHub UI

Workflow steps:
1. Checkout repository
2. Set up Python 3.x
3. Install dependencies from `requirements-docs.txt`
4. Deploy to GitHub Pages using `mkdocs gh-deploy`

## Troubleshooting

### "Page not found" errors

Ensure all links use relative paths:
```markdown
[Installation](../getting-started/installation.md)  # ✓ Correct
[Installation](/getting-started/installation.md)   # ✗ Incorrect
```

### MathJax not rendering

Check that `javascripts/mathjax.js` is configured in `mkdocs.yml`:

```yaml
extra_javascript:
  - javascripts/mathjax.js
  - https://polyfill.io/v3/polyfill.min.js?features=es6
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js
```

### Build fails with "Config file not found"

Ensure `mkdocs.yml` is in the repository root, not in `docs/`.

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Markdown Guide](https://www.markdownguide.org/)
- [MathJax Documentation](https://www.mathjax.org/)

## Contributing

When contributing documentation:

1. Follow the existing structure and style
2. Test locally with `mkdocs serve` before committing
3. Use clear, concise language
4. Include code examples where appropriate
5. Add cross-references to related pages
6. Keep line length reasonable (80-100 characters) for readability
