#!/usr/bin/env python3
"""
Result access example: How to access and interpret evaluation results.
"""
# Assumes you have already run an evaluation and have a result object

# [snippet-start]
# Access evaluation results
# result = evaluate(eval_cfg=eval_config, target_cfg=target_config)

# Access task-level metrics
task_result = result.tasks['mmlu_pro']
accuracy = task_result.metrics['acc'].scores['acc'].value
print(f"MMLU Pro Accuracy: {accuracy:.2%}")

# Access metrics with statistics
acc_metric = task_result.metrics['acc']
acc = acc_metric.scores['acc'].value
stderr = acc_metric.scores['acc'].stats.stderr
print(f"Accuracy: {acc:.3f} ± {stderr:.3f}")
# [snippet-end]

