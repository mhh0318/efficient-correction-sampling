# Editing the project page

The project page has its own source: **`docs/page.md`**. It does not read or synchronize with the repository README.

The published URL is cited by the arXiv paper. **Edit the separate draft workspace**, not the published page:

```text
/global/home/users/donghoon/correction-training-draft/docs/page.md
```

That workspace uses the `draft` branch. The published `main` branch deploys only when explicitly requested; pushes do not automatically update the live page.

- Edit the YAML block at the top for the title, authors, affiliations, navigation and resource links.
- Edit the Markdown below it for paragraphs, headings, tables, image paths and captions.
- Put images in `docs/assets/`, then use `![Description](assets/filename.png)` on its own line.
- The top sample figure uses SVG on desktop and JPEG on mobile. Its `mobile-srcset` attribute selects 768px or 1280px based on screen width and pixel density; clicking opens the SVG.
- Images below the top section load lazily. The first sample figure loads immediately.
- The `::::` section boundaries and `:::` paragraph boundaries retain the existing page layout.
- `docs/template.html` contains the page shell; `docs/project.css` controls styling.
- `docs/index.html` is generated. Editing it directly does not change the Markdown source.

## Preview locally

Pandoc and Python 3 are required. Pandoc is already installed on the current server.

From the draft repository root:

```bash
cd /global/home/users/donghoon/correction-training-draft
python3 scripts/build_page.py
python3 -m http.server 8850 --bind 127.0.0.1 --directory docs
```

Forward port 8850 in remote VSCode and open `http://localhost:8850/`. Rebuild after Markdown edits, then refresh the browser. If the preview server is already running, only the build command is needed.

## Save draft changes

```bash
git add docs/page.md docs/assets
git commit -m "Update project page"
git push origin draft
```

Saving or pushing the draft does not change the published page. Publication is a separate step after review: approved changes must be moved to `main`, then the **Build and deploy project page** Actions workflow must be run manually on `main`. Do not run it while reviewing a draft.

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
