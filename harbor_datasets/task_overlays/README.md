# Harbor task overlays

Vendored fixes to Harbor task *environments*, applied file-over-file on top of
downloaded task directories by `nemo_evaluator.environments.harbor`
(`_apply_task_overlays`, called after every dataset download or cache hit).

Layout: `<dataset-name>@<version>/<task-name>/<files...>` — dataset name `/`
sanitized to `__`. Only ship environment/config files here (Dockerfile, setup
scripts, task.toml); never test content, instructions, or solutions — those
are benchmark data and stay in the upstream dataset repo.

Current overlays:

- `swebench-verified@1.0/psf__requests-{1724,1766,1921,2317}` — hermetic
  httpbin: these four suites' graded tests dial httpbin.org live, which sheds
  load for ~10 hours every day
  ([SWE-bench/SWE-bench#622](https://github.com/SWE-bench/SWE-bench/issues/622)),
  failing correct patches. The overlay bakes an isolated local httpbin into
  the task image and a `[verifier] setup_script` boots it (with a hosts-level
  redirect of `httpbin.org` and a locally-trusted CA for the https tests)
  before `tests/test.sh` runs. Test files, graded lists, and the verifier
  procedure are untouched.
