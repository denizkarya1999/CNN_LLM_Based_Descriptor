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
    choices = {"no_action": {"kind": "no_action", "detection_id": None}}
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
    step = choices[choice["next_step"]] if choice else choices["no_action"]
    decision = "No Action" if step["kind"] == "no_action" else "Take Action"
    classification = cfg.task == "Image classification"
    if classification:
        facts = "Classification candidates: " + (
            "; ".join(f"{item['label']} ({item['confidence_percent']:g}%)" for item in evidence)
            or "none above the selected threshold.")
        context = "Unconfirmed. Classification does not locate or count objects."
        advice = "Object-specific actions are unavailable in classification mode."
        reason = "Classification provides candidate labels without bounding boxes, so it cannot support object-specific actions."
    elif not evidence:
        facts = "No objects detected by YOLO above the selected threshold."
        context = "Unconfirmed; no detections does not mean the scene is empty."
        advice = "No object-based action can be recommended from this frame."
        reason = f"YOLO supplied no valid object boxes at or above {cfg.confidence * 100:g}% detector confidence."
    else:
        counts = Counter(item["label"] for item in evidence)
        facts = "; ".join(f"{count} × {label}" for label, count in counts.items()) + "."
        context = {"indoor": "appears indoors (model estimate).",
                   "outdoor": "appears outdoors (model estimate).",
                   "unclear": "Unclear."}[environment]
        target = next((item for item in evidence if item["id"] == step["detection_id"]), None)
        if target is None:
            advice = "No specific action is suggested for the listed detections."
            reason = "The listed detections establish object presence only; object presence alone does not require intervention."
        else:
            name = f"{target['label']} at {target['position']} in the image"
            advice = {
                "verify": f"Confirm the possible {name} with another view before acting on this low-score detection.",
                "inspect": f"Check the detected {name} directly before deciding how to interact with it.",
                "give_space": f"If moving nearby, leave space around the detected {name}; check its current position directly.",
                "handle_carefully": f"If handling the detected {name}, inspect it first and handle it carefully.",
                "check_signal": f"Check the actual state of the detected {name} before deciding what to do.",
            }[step["kind"]]
            basis = {
                "verify": "This is below the 50% review threshold, so the label needs confirmation.",
                "inspect": "Detection alone does not establish its condition or how to interact with it.",
                "give_space": "Leaving space is a precaution if moving nearby; distance and movement are not established by the detection.",
                "handle_carefully": "This detected class calls for care if handled; its current condition is unconfirmed.",
                "check_signal": "The box and class label do not establish its current state.",
            }[step["kind"]]
            reason = f"YOLO reports {name} with {target['confidence_percent']:g}% detector confidence. {basis}"
        if any(item["confidence"] < LOW_SCORE for item in evidence):
            facts += " Detections below 50% are tentative."
    if total > len(evidence):
        facts += f" {total - len(evidence)} additional predictions omitted; counts cover listed predictions only."
    if status == "fallback":
        context = "Unconfirmed."
        advice = "No action recommendation is available."
        reason = "The model answer failed validation, so only YOLO facts are shown."
    scores = "; ".join(
        f"{item['label']} ({item.get('position', 'classification')}, {item['confidence_percent']:g}%)"
        for item in evidence)
    summary = "YOLO scores: " + (scores or "No predictions above the selected threshold.")
    detections = facts + (f"\n{summary}" if evidence and not classification else "")
    return {
        "text": f"Detections: {detections}\n\nEnvironment: {context}\n\nDecision: {decision}\n{advice}\n\nReason: {reason}",
        "grounding_summary": summary,
        "model": cfg.describe_model,
        "evidence": evidence, "total_predictions": total, "frame_size": [width, height],
        "grounding_status": status, "environment": environment,
        "action_decision": decision,
        "reason": reason,
        "suggested_action": {**step, "text": advice},
    }
