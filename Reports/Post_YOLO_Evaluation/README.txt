POST-YOLO ACTION EVALUATION - OCTOBER 29, 2026

Post_YOLO_Evaluation.pdf explains all nine implemented methods, their actual
responses, and a practical recommendation order for efficient actions.
Ranking is an engineering judgment with explicit tradeoffs, not a
universal accuracy or throughput leaderboard. All timings and decisions
come from the original benchmark; no model code was changed for this report.

results.csv: individual calls, repetition/phase, latency, action/status,
reference policies, image IDs and detected/considered object counts.
summary.json: aggregate timing and both policy-agreement scores.
response_examples.json: per-method response counts and recorded outputs
for four public scenes, including evidence and review-plan steps.
search_quality.json: 100 small synthetic graphs per search algorithm,
compared with independently enumerated optimal costs.
stress.json: separate scaling tests up to 1,000 synthetic review targets.
policy_checks.json: 423 documented-policy checks per classical method.

Scope: 24 images from COCO val2017 across eight object-based groups;
one public dataset with varied scenes, not a multi-dataset evaluation.
Dataset: https://cocodataset.org/#download
Tested source commit: 61c9a7f073945b803a0e71c4521ef3ad490c6b23

Scores compare results with policies derived from object annotations.
They are NOT independently human-labeled robot action accuracy.
Fallback LLM responses count as unavailable/incorrect, even if the GUI
shows No Action. Repetitions are not independent accuracy samples.
