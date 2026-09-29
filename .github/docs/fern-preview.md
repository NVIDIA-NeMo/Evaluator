# Fern preview operations

One `Preview Fern Docs` workflow runs on approved `pull-request/<number>` mirror
pushes. There is no artifact collector or `workflow_run` publisher. A change gate
compares the mirror with the PR base and selects `docs/**` and
`.github/workflows/fern-docs-ci.yml`, matching `Fern docs (check)`.

Mirror creation must remain restricted to the repository's approval automation:
the workflow definition itself comes from the approved commit. Direct fork
pushes do not trigger hosted previews, and the preview job additionally rejects
fork PRs. Forks retain the unchanged token-free `Fern docs (check)` workflow.
The publisher verifies the numeric mirror ref, current open PR, repository, and
exact head SHA using GitHub's API, then rechecks before commenting.

The publisher installs Fern **5.29.0** outside the PR checkout, with npm lifecycle
scripts disabled. The helper, navigation, components, and docs configuration come
from `main`; only Markdown/MDX and document assets come from the mirror. CLI
configuration is fixed to organization `nvidia` and the pinned version. Navigation
and tooling changes must land on `main` before affecting previews. New pages must
be reachable through trusted navigation. No library generation is required here.

`DOCS_FERN_TOKEN` is scoped to the fixed CLI publish step. Package scripts are not
executed. Page-link API calls are omitted, so the emitted URL never receives the
token. Displayed URLs must use HTTPS on `nvidia-preview-*.docs.buildwithfern.com`,
without credentials, port overrides, query, or fragment. Only the bot's marked
preview comment is updated.

There is no `PUBLISH_FERN` gate. Live variable inspection was denied to the
authoring identity; secret availability is not inferred from source. Docs owners
must confirm configuration and a successful same-repository preview/comment after
rollout. Operational owner follow-up is tracked separately.

Offline regression command:

```sh
python3 -m unittest discover -s .github/scripts -p 'test_fern_preview.py'
```

Fixtures cover mirror identity, stale/fork/closed PR rejection, non-executable
package/script inputs, trusted configuration, deleted pages, symlinks, and URL
validation. They do not publish or use credentials.
