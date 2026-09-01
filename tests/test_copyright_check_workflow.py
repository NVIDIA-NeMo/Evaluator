from pathlib import Path


def test_copyright_check_summary_has_required_env():
    workflow = Path(".github/workflows/copyright-check.yml").read_text()
    summary = workflow.split("copyright-check-summary:", 1)[1]
    step = summary.split("- name: Result", 1)[1].split("run: |", 1)[0]

    assert "GH_TOKEN:" in step, "Result step must export GH_TOKEN for the gh CLI"
    assert "SKIPPING_IS_ALLOWED:" in step, "Result step must define SKIPPING_IS_ALLOWED"

    run = summary.split("- name: Result", 1)[1].split("run: |", 1)[1]
    assert (
        "${SKIPPING_IS_ALLOWED:-false}" in run or '"$SKIPPING_IS_ALLOWED"' in run
    ), "Script should safely reference SKIPPING_IS_ALLOWED"
