"""Uniform-cost search: rank frontier states by accumulated cost g."""
from .priority import priority_search


def solve(context):
    return priority_search(context, lambda g, h: g)
