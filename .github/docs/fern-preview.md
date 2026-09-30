# Fern previews

Like Voice-Agent, `Preview Fern Docs` runs on approved `pull-request/<number>`
mirror pushes, with a changed-file gate and one preview job; no artifact handoff.
It selects `docs/**` and `.github/workflows/fern-docs-ci.yml`, matching the
unchanged token-free `Fern docs (check)` workflow.

- Mirror creation and workflow changes require approval: the workflow comes from
  the mirrored commit. Hosted previews accept only current, open same-repository
  PRs; forks retain token-free checks. There is no `PUBLISH_FERN` gate.
- Fern **5.29.0** is installed outside the checkout with npm lifecycle scripts
  disabled. Only document content is copied from the PR, without symlinks.
  Navigation, components and other configuration come from `main`; organization
  and CLI version are fixed in the workflow. Navigation changes must land on
  `main` to appear in previews. No library generation or package scripts run.
- `DOCS_FERN_TOKEN` is available only to the fixed CLI publish step. The comment
  accepts only HTTPS NVIDIA Fern preview URLs. Direct page links are omitted;
  no token is sent to the emitted URL. The PR head is rechecked before commenting.

Docs owners must confirm token availability and a normal preview/comment after
rollout. Token rotation and past-run review remain private owner follow-up;
this change does not perform either operation.
