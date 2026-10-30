"""Load and validate the shared label/action and scene knowledge base."""
import json
import math
from pathlib import Path
from .scenes import SETTINGS

MODEL_PATH = Path(__file__).with_name("scene_model.json")


def load_scene_model(path=MODEL_PATH):
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    labels, scenes = model["labels"], model["scenes"]
    if len(labels) != len(set(labels)) or set(scenes) != set(SETTINGS):
        raise ValueError("Knowledge base needs unique labels and every supported setting.")
    if set(model["label_actions"]) != set(labels):
        raise ValueError("Every knowledge-base label needs a decision association.")
    for data in scenes.values():
        if set(data["likelihoods"]) != set(labels):
            raise ValueError("Every setting must define likelihoods for every knowledge-base label.")
        for value in (data["prior"], data["default_likelihood"], *data["likelihoods"].values()):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError("Scene priors and likelihoods must be strictly between zero and one.")
    if not math.isclose(sum(data["prior"] for data in scenes.values()), 1, abs_tol=1e-8):
        raise ValueError("Scene priors must sum to one.")
    if len({data["default_likelihood"] for data in scenes.values()}) != 1:
        raise ValueError("Custom labels need equal neutral likelihoods across settings.")
    if not 0 <= model["low_score_weight"] <= 1:
        raise ValueError("Low-score evidence weight must be between zero and one.")
    review = model["review_model"]
    if not 0 < review["threshold"] < 1 or any(not 0 <= p <= 1 for p in review["probabilities"].values()):
        raise ValueError("Review probabilities and threshold must be valid probabilities.")
    if set(review["probabilities"]) != {"no_action", "verify", "check_signal", "handle_carefully"}:
        raise ValueError("Unsupported review probability table.")
    if review["probabilities"]["no_action"] >= review["threshold"]:
        raise ValueError("Presence-only review probability must stay below the recommendation threshold.")
    if not set(model["label_actions"].values()) <= {"no_action", "check_signal", "handle_carefully"}:
        raise ValueError("Unsupported label action in knowledge base.")
    # Only the matching detected classes may receive these specific suggestions.
    for label, kind in model["label_actions"].items():
        if kind == "check_signal" and label not in {"traffic light", "stop sign"}:
            raise ValueError("Signal advice requires a signal label.")
        if kind == "handle_carefully" and label not in {"knife", "scissors"}:
            raise ValueError("Handling advice requires a supported detected label.")
    return model
