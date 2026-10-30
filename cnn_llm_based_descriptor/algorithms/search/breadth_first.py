"""Breadth-first search: visit the shallowest frontier state first."""
from collections import deque


def solve(context):
    frontier, seen = deque([context.initial]), {(0, -1)}
    while frontier:
        node = frontier.popleft()
        context.visit(node, len(frontier))
        if node[0] == context.goal:
            return context.result(node)
        candidates = list(context.children(node))
        for child in candidates:
            if child[:2] not in seen:
                seen.add(child[:2]); frontier.append(child)
                context.pushed(frontier)
    raise RuntimeError("No review plan found.")
