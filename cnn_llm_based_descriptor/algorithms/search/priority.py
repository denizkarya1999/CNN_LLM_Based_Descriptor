"""Heap frontier and duplicate handling shared by cost/heuristic searches."""
import heapq
import itertools
import math
from .state import heuristic


def priority_search(context, rank, close_once=False):
    counter = itertools.count()
    frontier = [(0.0, next(counter), context.initial)]
    best, closed = {(0, -1): 0.0}, set()
    while frontier:
        _, _, node = heapq.heappop(frontier)
        state = node[:2]
        if node[2] > best[state] or (close_once and state in closed):
            continue
        closed.add(state)
        context.visit(node, len(frontier))
        if node[0] == context.goal:
            return context.result(node)
        for child in context.children(node):
            if close_once and child[:2] in closed:
                continue
            if child[2] >= best.get(child[:2], math.inf):
                continue
            best[child[:2]] = child[2]
            h = heuristic(context.targets, child[0], context.targets[child[1]]["center"])
            heapq.heappush(frontier, (rank(child[2], h), next(counter), child))
            context.pushed(frontier)
    raise RuntimeError("No review plan found.")
