# Editing the project page

The project page has its own source: **`docs/page.md`**. It does not read or synchronize with the repository README.

- Edit the YAML block at the top for the title, authors, affiliations, navigation and resource links.
- Edit the Markdown below it for paragraphs, headings, tables, image paths and captions.
- Put images in `docs/assets/`, then use `![Description](assets/filename.png)` on its own line.
- The `::::` section boundaries and `:::` paragraph boundaries retain the existing page layout.
- `docs/template.html` contains the page shell; `docs/project.css` controls styling.
- `docs/index.html` is generated. Editing it directly does not change the Markdown source.

## Preview locally

Pandoc and Python 3 are required. Pandoc is already installed on the current server.

From the repository root:

```bash
python3 scripts/build_page.py
python3 -m http.server 8847 --bind 127.0.0.1 --directory docs
```

Forward port 8847 in remote VSCode and open `http://localhost:8847/`. Rebuild after Markdown edits, then refresh the browser. If the preview server is already running, only the build command is needed.

## Publish

```bash
git add docs/page.md docs/assets
git commit -m "Update project page"
git push origin main
```

GitHub Actions rebuilds the HTML from `page.md` and deploys it automatically. It does not require a local rebuild or a separate commit of generated HTML. Template and CSS changes also trigger deployment.

The public URL remains `https://sunovivid.github.io/correction-training/`.

## Add a section

```markdown
:::: {.content-section data-section="New section"}

## New section {#new-section}

Your paragraph here.

![Image description](assets/example.png)

::::
```

To link it from the top navigation, add a `navigation` entry in the YAML block with `href: "#new-section"`.

## Expandable gallery

Keep blank lines around the Markdown inside `<details>`:

```markdown
<details>
<summary>More examples</summary>

![Example](assets/example.png)

</details>
```
