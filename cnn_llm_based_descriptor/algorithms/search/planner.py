"""Dispatch each batch to a named strategy and assemble the full review plan."""
from ...methods import SEARCH_ALGORITHMS, DEFAULT_SEARCH
from .state import SearchContext, TRACE_LIMIT
from . import breadth_first, depth_first, iterative_deepening, uniform_cost, greedy_best_first, astar

BATCH_SIZE = 5
STRATEGIES = dict(zip(SEARCH_ALGORITHMS, (
    breadth_first.solve, depth_first.solve, iterative_deepening.solve,
    uniform_cost.solve, greedy_best_first.solve, astar.solve,
)))


def search_batch(targets, algorithm, start=(.5, .5), audit=None):
    if algorithm not in STRATEGIES:
        raise ValueError("Choose a supported search algorithm.")
    return STRATEGIES[algorithm](SearchContext(targets, algorithm, start, audit))


def plan_reviews(targets, algorithm=DEFAULT_SEARCH):
    if algorithm not in SEARCH_ALGORITHMS:
        raise ValueError("Choose a supported search algorithm.")
    steps, batches = [], []
    cost, expanded, origin = 0.0, 0, (.5, .5)
    traversal, generated, peak = [], 0, 0
    for offset in range(0, len(targets), BATCH_SIZE):
        group = targets[offset:offset+BATCH_SIZE]
        audit = {}
        path, group_cost, count, last = search_batch(group, algorithm, origin, audit)
        generated += audit["generated_states"]
        peak = max(peak, audit["peak_frontier"])
        traversal.extend({**entry, "batch": len(batches)+1}
                         for entry in audit["traversal"][:max(0, TRACE_LIMIT-len(traversal))])
        steps.extend(path); cost += group_cost; expanded += count
        batches.append({"targets": [t["id"] for t in group], "cost": group_cost,
                        "expanded_states": count})
        origin = group[last]["center"]
    return {"algorithm": algorithm, "steps": steps, "cost": cost,
            "expanded_states": expanded, "targets": [t["id"] for t in targets],
            "traversal": traversal, "trace_omitted": expanded-len(traversal),
            "generated_states": generated, "peak_frontier": peak,
            "omitted_targets": 0, "batches": batches, "batch_size": BATCH_SIZE,
            "globally_cost_optimal": algorithm in (SEARCH_ALGORITHMS[3], SEARCH_ALGORITHMS[5]) and len(batches) <= 1,
            "goal": "Plan a review suggestion for every rule-triggered target; no actions executed.",
            "cost_basis": "Illustrative effort plus Manhattan attention shift between normalized image box centers, not physical distance or risk.",
            "optimality_scope": "UCS/A* minimize cost within each batch; multiple batches are not globally optimal."}
