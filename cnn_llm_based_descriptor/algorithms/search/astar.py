"""A* search: rank by accumulated cost g plus admissible heuristic h."""
from .priority import priority_search


def solve(context):
    return priority_search(context, lambda g, h: g + h)
