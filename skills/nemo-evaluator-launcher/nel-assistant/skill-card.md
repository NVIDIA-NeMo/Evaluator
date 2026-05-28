## Description: <br>
Interactive config wizard for NeMo Evaluator Launcher (NEL) that guides users through creating, configuring, and running production-ready evaluation configurations. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who need to create NeMo Evaluator Launcher YAML configurations interactively, set up model evaluations from scratch, or modify existing NEL configs for deployment, tasks, multi-node, and interceptors. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NeMo Evaluator GitHub Issues](https://github.com/NVIDIA-NeMo/Evaluator/issues) <br>
- [NeMo Evaluator GitHub Discussions](https://github.com/NVIDIA-NeMo/Evaluator/discussions) <br>
- [NeMo Evaluator Interceptors Documentation](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/interceptors/index.html) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks and YAML configurations] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Tasks: <br>
NVSkills-Eval 3-Tier Evaluation (external profile). Tier 1 static validation: 9 checks, PASS. Tier 2 deduplication: 2 checks, PASS. Overall verdict: PASS. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>



## Skill Version(s): <br>
0.2.6 (source: git tag nemo-evaluator-launcher-v0.2.6) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
