## Description: <br>
Run, monitor, analyze, and debug LLM evaluations via nemo-evaluator-launcher. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to run, monitor, debug, and analyze LLM evaluations on remote clusters via the nemo-evaluator-launcher CLI. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [run-evaluation.md](references/run-evaluation.md) <br>
- [check-progress.md](references/check-progress.md) <br>
- [analyze-results.md](references/analyze-results.md) <br>
- [debug-failed-runs.md](references/debug-failed-runs.md) <br>
- [swebench-general-info.md](references/benchmarks/swebench-general-info.md) <br>
- [terminal-bench-general-info.md](references/benchmarks/terminal-bench-general-info.md) <br>
- [terminal-bench-trace-analysis.md](references/benchmarks/terminal-bench-trace-analysis.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Analysis, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Tasks: <br>
3-Tier NVSkills-Eval evaluation (external profile). Tier 1: 9 static validation checks (14 findings). Tier 2: deduplication checks (0 findings). Overall verdict: PASS. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>



## Skill Version(s): <br>
nemo-evaluator-launcher-v0.2.6-12-g36975fa7 (source: git tag) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
