# Documentation Development

The user documentation is built with MkDocs and Material for MkDocs.

## Build

```bash
pip install -r requirements-docs.txt
./build_docs.sh
```

For live reload:

```bash
mkdocs serve
```

## Structure

The canonical user pages live directly under `docs/` and are listed in
`mkdocs.yml`:

```text
docs/
  index.md
  quickstart.md
  installation.md
  parameters.md
  runtime-arguments.md
  gridding.md
  mpi.md
  conversion.md
```

Add a new page to `mkdocs.yml` when it should appear in site navigation. Use
paths relative to the current Markdown file, for example:

```markdown
[Installation](installation.md)
```

Run `mkdocs build --strict` before committing to catch invalid navigation and
link warnings.
