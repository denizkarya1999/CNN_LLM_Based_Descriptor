VISUAL APPENDIX - PAGES 7-12, ORDERED BY EFFICIENCY

Page 7: rank 1 - general robot plans.
Page 8: rank 2 (tie) - A* and UCS search trees.
Page 9: rank 3 - probability calculations and review selection.
Page 10: rank 4 - greedy search; rank 5 (tie) - BFS.
Page 11: rank 5 (tie) - IDS; rank 6 - DFS.
Page 12: rank 7 - LLM processing, validation, explanation and fallback.

Visuals follow the efficiency recommendation on page 1: usable actions,
policy fit, review-plan cost and computation. This is not a latency-only
sort. Equal recommendation tiers retain the same rank number.

The main example is COCO val2017 image 43314 (skiing). The LLM rejection
example is image 555050 (street intersection). All recorded values come
from warm repetition 0 of the existing benchmark, tested source commit
61c9a7f073945b803a0e71c4521ef3ad490c6b23. No models were rerun for these pages.

visualization_data.json contains the extracted evidence and complete small
search traces, probability calculations, robot-rule evaluations, selected
LLM responses, and SHA-256 hashes of the original benchmark JSONL files.
The larger raw logs are not needed to regenerate these diagrams.

The small search tree includes both possible complete review sequences.
Gray nodes are possible states not visited before that algorithm stopped;
some were generated but never visited. All node costs occur in the saved
traces of at least one method. Blue badges are actual visit numbers; green
edges identify the returned path. IDS can revisit a node at a new limit.
The plots describe this two-target example, not every possible search tree.
Search costs are illustrative review effort, not milliseconds or distance.

The LLM drawing represents application inputs, outputs and validation.
It is not a trace of hidden model reasoning. Solid paths illustrate the two
saved examples; the dashed explanation-retry path is implemented behavior
and was not taken in the skiing call. Its explanation was accepted by the
application, which is not independent evidence that its content is correct.

REBUILD

Dependencies: Python 3, reportlab, pypdf (report tooling only).
From the project root:

python Reports/Post_YOLO_Evaluation/visuals/build_visual_appendix.py

The generator reads visualization_data.json, preserves the report's first
six pages, and replaces the six visual pages. Repeated runs do not append
duplicate pages. Use --report PATH and --output PATH to work on another copy.
The PDF uses vector lines, boxes, text and bars so the diagrams stay sharp
when zoomed. After editing the diagrams, render and visually check every
changed page before replacing the published report.

VERIFICATION PERFORMED

Matched all six saved search traces to the original benchmark records and
recalculated each path's cost from normalized box centers. Independently
normalized all 20 scene scores, checked the 12 robot rules and two LLM
response routes, confirmed that original report pages 1-6 are unchanged,
and rendered/reviewed all six visual pages.

The ranked revision also checks that all nine diagrams follow ranks
1, 2, 2, 3, 4, 5, 5, 6, 7, with source measurements unchanged.
