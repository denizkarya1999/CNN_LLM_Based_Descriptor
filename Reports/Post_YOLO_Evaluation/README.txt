POST-YOLO ACTION EVALUATION - OCTOBER 29, 2026

Post_YOLO_Evaluation.pdf is an eleven-page report covering all nine implemented
methods: how each processes YOLO evidence, its recorded responses, numerical
performance against two reference policies, and an efficiency recommendation.
Ranking is an engineering judgment with explicit tradeoffs, not a universal
accuracy or throughput leaderboard. Measurements remain from the original
benchmark; revisions add derived metrics, reference documentation and visual
explanations without rerunning models.

VISUAL PAGE GUIDE

Pages 7-8: search trees for BFS, DFS, IDS, UCS, greedy best-first and A*.
Page 9: probability calculations and review selection.
Page 10: robot-plan counts, rule matches and selected recommendation.
Page 11: LLM inputs, decision, validation, explanation and fallback.

The drawings use recorded examples; search nodes show actual visit numbers,
costs and selected paths. visuals/ contains extracted source data and a
rebuild script. Original metric and reference pages 1-6 are preserved.

FILES

performance_metrics.csv / .json: per-method accuracy, precision, recall, F1,
macro-F1, balanced accuracy, decision availability, confusion counts, 95%
Wilson accuracy intervals, and median / 95th-percentile latency. JSON also
includes constant-action baselines and metric definitions. All metric rates
are percentages (0-100); latency columns are milliseconds.

ground_truth.csv: the expected review action, robot action, primary robot
plan, object counts by label, and original image URL for each of 24 images.
ground_truth.json: the same references plus original COCO annotation IDs,
category IDs, labels, bounding boxes, source URL/member SHA-256, and policies.
These annotations were checked directly against instances_val2017.json in
the official COCO trainval2017 annotation archive.

results.csv: original individual calls, repetition/phase, latency, action,
status, reference policies, image IDs and detected/considered object counts.
summary.json: original aggregate timing and both policy-agreement scores.
response_examples.json: per-method response counts and recorded outputs
for four public scenes, including evidence and review-plan steps.
search_quality.json: 100 small synthetic graphs per search algorithm,
compared with independently enumerated optimal costs.
stress.json: separate scaling tests up to 1,000 synthetic review targets.
policy_checks.json: 423 documented-policy checks per classical method.

GROUND TRUTH AND SCOPE

Official COCO annotations provide human-labeled OBJECT ground truth.
Action references are obtained by applying explicitly documented policies
to those annotations, without using algorithm outputs or LLM judgments.
They are NOT independently human-labeled robot action truth. An uncertainty
review can disagree with these references because annotations are treated
as confirmed while low-confidence YOLO predictions can trigger verification.

Scope: 24 images from COCO val2017 across eight object-based groups, three
per group, selected with seed 20261029 before model outputs. This is one
public dataset with varied scenes, not a multi-dataset evaluation. Crowd
annotations (iscrowd=1) are excluded. Positive class: Take Action.
Review references: 7 positive / 17 negative.
Robot references: 21 positive / 3 negative.
Dataset: https://cocodataset.org/#download
Tested source commit: 61c9a7f073945b803a0e71c4521ef3ad490c6b23

METRIC CONVENTIONS

Accuracy uses only phase=warm, repetition=0: 24 distinct images per method.
Repeated runs contribute timing data, not extra independent accuracy cases.
Latency uses all warm repetitions, excluding no-detection LLM bypasses:
120 calls per classical method; 46 LLM inference calls out of 48 total calls.
First calls are excluded from these timing distributions and saved separately.
Timing includes input preparation, validation and explanation, but excludes
YOLO inference and client construction. Only the LLM also reads image pixels;
it receives at most ten detections and can further truncate its prompt.

TP: valid Action on a positive reference.
TN: valid No Action on a negative reference.
FP: valid Action on a negative reference.
FN: valid No Action on a positive reference.
AP / AN: unavailable result on a positive / negative reference.
N = TP + TN + FP + FN + AP + AN.

Accuracy = (TP + TN) / N.
Precision = TP / (TP + FP); undefined precision is null/blank.
Recall = TP / (TP + FN + AP).
Specificity = TN / (TN + FP + AN).
Positive F1 = 2TP / (2TP + FP + FN + AP).
Negative F1 = 2TN / (2TN + FN + FP + AN).
Macro-F1 = mean of positive and negative F1.
Balanced accuracy = mean of recall and specificity.
Decision availability = (N - AP - AN) / N.

Rejected or fallback LLM responses count as unavailable and incorrect even
if the GUI displays No Action. No-detection bypasses are valid No Action.
Abstentions remain in accuracy/recall/F1 denominators as specified above;
precision considers only accepted Action outputs. The LLM's 100% precision
against robot references is based on just one accepted Action, with 4.8%
recall and 54.2% availability, and must not be read as strong recognition.

Search-plan cost is illustrative effort plus image-coordinate attention
movement, not physical distance or energy. UCS/A* optimality applies within
five-target batches. The 100 optimality fixtures have at most five targets;
the separate 1,000-target stress test measures latency with three repeats.
No energy or peak-memory comparison was measured. Small class supports and
one public dataset limit generalization; these scores do not prove safe or
correct physical intervention.
