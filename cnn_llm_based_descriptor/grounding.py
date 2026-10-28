"""Render only detector facts and validated, bounded model choices.

Free-form model text never reaches the caption or recording. The vision model
can estimate a broad setting and select a suggestion tied to a current box.
"""

from collections import Counter
import json

ENVIRONMENTS = ("indoor", "outdoor", "unclear")
LOW_SCORE = 0.5
SPACE_LABELS = frozenset({
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "chair", "bench", "couch", "dining table",
})


def action_choices(evidence):
    """Offer suggestions only for supplied boxes, with no inferred object state."""
    choices = {}
    for item in evidence:
        if "box_xyxy_pixels" not in item:
            continue
        kinds = ["verify"] if item["confidence"] < LOW_SCORE else ["inspect"]
        if item["confidence"] >= LOW_SCORE:
            label = item["label"].casefold()
            if label in SPACE_LABELS:
                kinds.append("give_space")
            if label in {"knife", "scissors"}:
                kinds.append("handle_carefully")
            if label in {"traffic light", "stop sign"}:
                kinds.append("check_signal")
        for kind in kinds:
            choices[f"{kind}:{item['id']}"] = {"kind": kind, "detection_id": item["id"]}
    return choices


def response_schema(choices):
    return {
        "type": "object",
        "properties": {
            "environment": {"type": "string", "enum": list(ENVIRONMENTS)},
            "next_step": {"type": "string", "enum": list(choices)},
        },
        "required": ["environment", "next_step"],
        "additionalProperties": False,
    }


def parse_choice(content, choices):
    """Reject the entire answer if it contains prose, extra fields or unknown IDs."""
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate model field")
            result[key] = value
        return result

    if not isinstance(content, str) or len(content) > 2000:
        raise ValueError("Invalid model choice")
    answer = json.loads(content, object_pairs_hook=unique_keys)
    if not isinstance(answer, dict) or set(answer) != {"environment", "next_step"}:
        raise ValueError("Unexpected model fields")
    if not isinstance(answer["environment"], str) or answer["environment"] not in ENVIRONMENTS:
        raise ValueError("Unsupported environment")
    if not isinstance(answer["next_step"], str) or answer["next_step"] not in choices:
        raise ValueError("Unsupported suggestion or detection ID")
    return answer


def render_description(evidence, total, width, height, cfg, choice=None, status="validated"):
    """All object words, counts, scores and locations originate in YOLO evidence."""
    choices = action_choices(evidence)
    environment = choice["environment"] if choice else "unclear"
    step = choices[choice["next_step"]] if choice else {"kind": "observe", "detection_id": None}
    classification = cfg.task == "Image classification"
    if classification:
        facts = "YOLO classification candidates: " + (
            "; ".join(f"{item['label']} ({item['confidence_percent']:g}%)" for item in evidence)
            or "none above the selected threshold.")
        context = "Classification does not locate or count objects."
        advice = "Switch to object detection for suggestions tied to detected objects."
    elif not evidence:
        facts = "YOLO detected no objects above the selected threshold."
        context = "The environment is unconfirmed; no detections does not mean the scene is empty."
        advice = "Check the view and capture another frame before deciding what to do."
    else:
        counts = Counter(item["label"] for item in evidence)
        facts = "YOLO detected: " + "; ".join(f"{count} × {label}" for label, count in counts.items()) + "."
        context = {"indoor": "Environment: appears indoors (model estimate).",
                   "outdoor": "Environment: appears outdoors (model estimate).",
                   "unclear": "Environment: unclear."}[environment]
        target = next((item for item in evidence if item["id"] == step["detection_id"]), None)
        if target is None:
            advice = "Review the listed detections and confirm the current view before deciding what to do."
        else:
            name = f"{target['label']} at {target['position']} in the image"
            advice = {
                "verify": f"Confirm the possible {name} with another view before acting on this low-score detection.",
                "inspect": f"Check the detected {name} directly before deciding how to interact with it.",
                "give_space": f"If moving nearby, leave space around the detected {name}; check its current position directly.",
                "handle_carefully": f"If handling the detected {name}, inspect it first and handle it carefully.",
                "check_signal": f"Check the actual state of the detected {name} before deciding what to do.",
            }[step["kind"]]
        if any(item["confidence"] < LOW_SCORE for item in evidence):
            facts += " Detections below 50% are tentative."
    if total > len(evidence):
        facts += f" {total - len(evidence)} additional predictions omitted; counts cover listed predictions only."
    if status == "fallback":
        context = "Environment: unconfirmed. The model answer failed validation; showing YOLO facts only."
    scores = "; ".join(
        f"{item['label']} ({item.get('position', 'classification')}, {item['confidence_percent']:g}%)"
        for item in evidence)
    return {
        "text": f"{facts}\n{context}\nSuggested next step: {advice}",
        "grounding_summary": "YOLO scores: " + (scores or "No predictions above the selected threshold."),
        "model": cfg.describe_model,
        "evidence": evidence, "total_predictions": total, "frame_size": [width, height],
        "grounding_status": status, "environment": environment,
        "suggested_action": {**step, "text": advice},
    }
