# Project page editing

- `https://sunovivid.github.io/correction-training/` is the published page cited by the arXiv paper. Routine draft work must not change or deploy that page.
- Keep preview code and rendered HTML on the `draft` branch, in `/global/home/users/donghoon/correction-training-draft`. The user continues editing `/global/home/users/donghoon/correction-training-public/docs/page.md`; the draft server reads that file directly via `.preview-source`. Do not substitute a stale copied Markdown file. The preview is independent from the repository README.
- Serve the draft privately at `http://localhost:8850/`. Keep draft indexing disabled.
- Preserve the user's draft text and unrelated local edits.
- Deployment of the published page requires an explicit user request to publish the draft there. Pushing a draft or editing a preview is not authorization to update the published page.
- The main Pages workflow is manual-only and only accepts `main`. Do not re-enable automatic live deployment as part of draft work.
