"""Label/score review rules shared by search and probability clients."""
from .grounding import LOW_SCORE


def assess_detections(evidence, model):
    """Every eligible box gets a decision association, even if it needs no action."""
    assessments = []
    for item in evidence:
        if "box_xyxy_pixels" not in item:
            continue
        label = item["label"].casefold()
        kind = "verify" if item["confidence"] < LOW_SCORE else model["label_actions"].get(label, "no_action")
        rule = {"verify": "low_score", "check_signal": "signal_state_unconfirmed",
                "handle_carefully": "care_if_handled", "no_action": "presence_only"}[kind]
        assessments.append({"detection_id": item["id"], "label": item["label"], "kind": kind,
                            "rule": rule, "confidence": item["confidence"],
                            "knowledge_base_label": label in model["label_actions"]})
    return assessments
