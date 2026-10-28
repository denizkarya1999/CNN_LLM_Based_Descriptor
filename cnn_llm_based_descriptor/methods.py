"""Names shared by settings, descriptor output, and saved metadata."""

DESCRIBE_BACKENDS = ("LLM", "Search Algorithm", "Probability Algorithm")
SEARCH_ALGORITHMS = (
    "Breadth-First Search (BFS)",
    "Depth-First Search (DFS)",
    "Iterative Deepening Search (IDS)",
    "Uniform-Cost Search (UCS)",
    "Greedy Best-First Search (GBFS)",
    "A* Search",
)
PROBABILITY_ALGORITHM = "Inference by Enumeration"
DEFAULT_SEARCH = "Uniform-Cost Search (UCS)"


def method_details(cfg):
    algorithm = (cfg.describe_model if cfg.describe_backend == "LLM" else
                 cfg.describe_search if cfg.describe_backend == "Search Algorithm" else
                 PROBABILITY_ALGORITHM)
    return {"descriptor_type": cfg.describe_backend, "algorithm": algorithm}


def latency_text(milliseconds):
    if milliseconds is None:
        return "Description inference latency: —"
    if milliseconds < .01:
        value = "<0.01 ms"
    elif milliseconds < 1000:
        value = f"{milliseconds:.2f} ms"
    else:
        value = f"{milliseconds / 1000:.3f} s"
    return "Description inference latency: " + value
