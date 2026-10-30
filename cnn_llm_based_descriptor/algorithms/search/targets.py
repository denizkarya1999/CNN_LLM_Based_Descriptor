"""Translate shared review-policy matches into search targets and costs."""
from ...methods import DEFAULT_SEARCH
from ..shared.grounding import action_choices
from ..shared.knowledge import load_scene_model
from ..shared.review_policy import assess_detections
from .planner import plan_reviews


def review_targets(evidence, assessments):
    offered, by_id = action_choices(evidence), {item["id"]: item for item in evidence}
    targets = []
    for assessment in assessments:
        kind, ident = assessment["kind"], assessment["detection_id"]
        if kind == "no_action":
            continue
        item = by_id[ident]
        kinds = ((kind, 1.2 if kind == "handle_carefully" else 1.0),)
        if kind != "verify":
            kinds += (("inspect", 1.6),)
        options = [(f"{action}:{ident}", cost) for action, cost in kinds if f"{action}:{ident}" in offered]
        if options:
            x1, y1, x2, y2 = item["box_xyxy_normalized"]
            targets.append({"id": ident, "rule": assessment["rule"], "options": options,
                            "center": ((x1+x2)/2, (y1+y2)/2)})
    return targets

def search_review_plan(evidence, algorithm=DEFAULT_SEARCH, model=None):
    model = model or load_scene_model()
    assessments = assess_detections(evidence, model)
    result = plan_reviews(review_targets(evidence, assessments), algorithm)
    result["assessments"] = assessments
    return result
