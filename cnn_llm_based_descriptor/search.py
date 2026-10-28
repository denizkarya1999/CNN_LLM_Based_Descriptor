"""Six genuine frontier strategies for image-review planning.

All targets are processed. Small sequential batches bound exponential work;
cost optimality of UCS/A* applies within a batch, not across batches.
"""

from collections import deque
import heapq
import itertools
import math

from .methods import SEARCH_ALGORITHMS, DEFAULT_SEARCH

BATCH_SIZE = 5
TRACE_LIMIT = 250


def distance(a, b):
    return sum(abs(x-y) for x, y in zip(a, b))


def heuristic(targets, mask, origin):
    """Admissible: required effort + the cheapest first attention shift.

Every completion must pay each remaining target's minimum effort and at least
the distance to one remaining target. Subsequent shifts are ignored.
"""
    remaining = [t for i, t in enumerate(targets) if not mask & (1 << i)]
    return (sum(min(cost for _, cost in t["options"]) for t in remaining)
            + min((distance(origin, t["center"]) for t in remaining), default=0))


def search_batch(targets, algorithm, start=(.5, .5), audit=None):
    """State=(planned target mask,last target); edges add a review suggestion."""
    if algorithm not in SEARCH_ALGORITHMS:
        raise ValueError("Choose a supported search algorithm.")
    goal = (1 << len(targets)) - 1
    # node=(mask,last,cost,path). The mask strictly grows, so there are no cycles.
    initial = (0, -1, 0.0, [])
    expanded = 0
    audit = audit if audit is not None else {}
    audit.update(traversal=[], generated_states=1, peak_frontier=1)

    def visit(node, frontier_size, limit=None):
        origin = start if node[1] == -1 else targets[node[1]]["center"]
        h = heuristic(targets, node[0], origin)
        priority = (node[2] if algorithm == SEARCH_ALGORITHMS[3] else
                    h if algorithm == SEARCH_ALGORITHMS[4] else
                    node[2] + h if algorithm == SEARCH_ALGORITHMS[5] else None)
        if len(audit["traversal"]) < TRACE_LIMIT:
            audit["traversal"].append({"visit": expanded, "path": [s["choice"] for s in node[3]],
                "depth": len(node[3]), "g": node[2], "h": h, "priority": priority,
                "goal": node[0] == goal, "depth_limit": limit})
        audit["peak_frontier"] = max(audit["peak_frontier"], frontier_size + 1)

    def pushed(frontier):
        audit["peak_frontier"] = max(audit["peak_frontier"], len(frontier))

    def children(node):
        mask, last, cost, path = node
        origin = start if last == -1 else targets[last]["center"]
        for i, target in enumerate(targets):
            if mask & (1 << i):
                continue
            shift = distance(origin, target["center"])
            for key, effort in target["options"]:
                edge = effort + shift
                step = {"choice": key, "detection_id": target["id"],
                        "rule": target["rule"], "cost": edge}
                audit["generated_states"] += 1
                yield mask | (1 << i), i, cost + edge, path + [step]

    if algorithm == SEARCH_ALGORITHMS[2]:
        # Re-run depth-limited DFS with a fresh frontier/visited set each time.
        for limit in range(len(targets) + 1):
            stack, seen = [initial], set()
            if limit:
                audit["generated_states"] += 1
            while stack:
                node = stack.pop()
                state = node[:2]
                if state in seen:
                    continue
                seen.add(state); expanded += 1
                visit(node, len(stack), limit)
                if node[0] == goal:
                    return node[3], node[2], expanded, node[1]
                if len(node[3]) < limit:
                    for child in reversed(list(children(node))):
                        stack.append(child); pushed(stack)
    elif algorithm in SEARCH_ALGORITHMS[:2]:
        frontier, seen = deque([initial]), {(0, -1)}
        while frontier:
            node = frontier.popleft() if algorithm == SEARCH_ALGORITHMS[0] else frontier.pop()
            expanded += 1
            visit(node, len(frontier))
            if node[0] == goal:
                return node[3], node[2], expanded, node[1]
            candidates = list(children(node))
            if algorithm == SEARCH_ALGORITHMS[1]:
                candidates.reverse()
            for child in candidates:
                if child[:2] not in seen:
                    seen.add(child[:2]); frontier.append(child)
                    pushed(frontier)
    else:
        counter = itertools.count()
        frontier = [(0.0, next(counter), initial)]
        best, closed = {(0, -1): 0.0}, set()
        greedy = algorithm == SEARCH_ALGORITHMS[4]
        while frontier:
            _, _, node = heapq.heappop(frontier)
            state = node[:2]
            if node[2] > best[state] or (greedy and state in closed):
                continue
            closed.add(state); expanded += 1
            visit(node, len(frontier))
            if node[0] == goal:
                return node[3], node[2], expanded, node[1]
            for child in children(node):
                if greedy and child[:2] in closed:
                    continue
                if child[2] >= best.get(child[:2], math.inf):
                    continue
                best[child[:2]] = child[2]
                h = heuristic(targets, child[0], targets[child[1]]["center"])
                priority = (child[2] if algorithm == SEARCH_ALGORITHMS[3] else
                            h if greedy else child[2] + h)
                heapq.heappush(frontier, (priority, next(counter), child))
                pushed(frontier)
    raise RuntimeError("No review plan found.")


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
