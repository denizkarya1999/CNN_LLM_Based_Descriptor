"""Shared state expansion, admissible heuristic and bounded traversal audit."""
from ...methods import SEARCH_ALGORITHMS

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

class SearchContext:
    def __init__(self, targets, algorithm, start, audit):
        self.targets, self.algorithm, self.start = targets, algorithm, start
        self.goal = (1 << len(targets)) - 1
        self.initial = (0, -1, 0.0, [])
        self.expanded = 0
        self.audit = audit if audit is not None else {}
        self.audit.update(traversal=[], generated_states=1, peak_frontier=1)

    def visit(self, node, frontier_size, limit=None):
        self.expanded += 1
        origin = self.start if node[1] == -1 else self.targets[node[1]]["center"]
        h = heuristic(self.targets, node[0], origin)
        priority = (node[2] if self.algorithm == SEARCH_ALGORITHMS[3] else
                    h if self.algorithm == SEARCH_ALGORITHMS[4] else
                    node[2] + h if self.algorithm == SEARCH_ALGORITHMS[5] else None)
        if len(self.audit["traversal"]) < TRACE_LIMIT:
            self.audit["traversal"].append({"visit": self.expanded, "path": [s["choice"] for s in node[3]],
                "depth": len(node[3]), "g": node[2], "h": h, "priority": priority,
                "goal": node[0] == self.goal, "depth_limit": limit})
        self.audit["peak_frontier"] = max(self.audit["peak_frontier"], frontier_size + 1)

    def pushed(self, frontier):
        self.audit["peak_frontier"] = max(self.audit["peak_frontier"], len(frontier))

    def children(self, node):
        mask, last, cost, path = node
        origin = self.start if last == -1 else self.targets[last]["center"]
        for i, target in enumerate(self.targets):
            if mask & (1 << i):
                continue
            shift = distance(origin, target["center"])
            for key, effort in target["options"]:
                edge = effort + shift
                step = {"choice": key, "detection_id": target["id"],
                        "rule": target["rule"], "cost": edge}
                self.audit["generated_states"] += 1
                yield mask | (1 << i), i, cost + edge, path + [step]

    def result(self, node):
        return node[3], node[2], self.expanded, node[1]
