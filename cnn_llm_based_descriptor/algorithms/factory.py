"""The single routing point from saved descriptor settings to an implementation."""


def create_descriptor_client(cfg):
    if cfg.describe_backend == "LLM":
        from .llm.client import OllamaVisionClient
        return OllamaVisionClient(cfg)
    if cfg.describe_backend == "Search Algorithm":
        from .search.client import SearchClient
        return SearchClient(cfg)
    if cfg.describe_backend == "Probability Algorithm":
        from .probability.client import ProbabilityClient
        return ProbabilityClient(cfg)
    if cfg.describe_backend == "General Algorithmic Approach":
        from .robot_plans.client import GeneralAlgorithmicClient
        return GeneralAlgorithmicClient(cfg)
    raise ValueError("Choose a supported descriptor type.")
