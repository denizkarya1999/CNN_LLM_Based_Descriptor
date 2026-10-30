"""Greedy best-first search: rank only by remaining-cost heuristic h."""
from .priority import priority_search


def solve(context):
    return priority_search(context, lambda g, h: h, close_once=True)
