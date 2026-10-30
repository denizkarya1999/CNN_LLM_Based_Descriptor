"""Detector-owned facts/actions, with evidence-referenced LLM explanations.

Explanation checks constrain references and known labels; they are not a proof
of the truth of arbitrary natural language. Keep its model provenance visible.
"""

from collections import Counter
import json
import re
from pathlib import Path
from functools import lru_cache
from .scenes import SETTINGS, environment_text
from ...methods import method_details, NO_ENVIRONMENT_BACKENDS

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


def response_schema(choices, explain=False):
    schema = {
        "type": "object",
        "properties": {
            "environment": {"type": "string", "enum": list(ENVIRONMENTS)},
            "setting": {"type": "string", "enum": list(SETTINGS)},
            "next_step": {"type": "string", "enum": list(choices)},
        },
        "required": ["environment", "setting", "next_step"],
        "additionalProperties": False,
    }
    if explain:
        schema["properties"]["reason"] = {"type": "string", "minLength": 20, "maxLength": 350}
        schema["required"].append("reason")
    return schema


def parse_choice(content, choices, evidence=None):
    """Validate bounded choices and, when requested, the cited explanation."""
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
    expected_keys = {"environment", "setting", "next_step"} | ({"reason"} if evidence is not None else set())
    if not isinstance(answer, dict) or set(answer) != expected_keys:
        raise ValueError("Unexpected model fields")
    if not isinstance(answer["environment"], str) or answer["environment"] not in ENVIRONMENTS:
        raise ValueError("Unsupported environment")
    if not isinstance(answer["setting"], str) or answer["setting"] not in SETTINGS:
        raise ValueError("Unsupported setting")
    expected = SETTINGS[answer["setting"]][1]
    if expected != "unclear" and answer["environment"] != expected:
        raise ValueError("Setting contradicts indoor/outdoor estimate")
    if not isinstance(answer["next_step"], str) or answer["next_step"] not in choices:
        raise ValueError("Unsupported suggestion or detection ID")
    if evidence is not None:
        validate_reason(answer["reason"], evidence, choices[answer["next_step"]]["detection_id"])
    return answer


@lru_cache(maxsize=1)
def known_object_labels():
    return json.loads(Path(__file__).with_name("scene_model.json").read_text())["labels"]


def validate_reason(reason, evidence, target):
    """Require explicit evidence references and reject known unsupported labels.

This intentionally does not claim semantic verification of generated prose.
"""
    if not isinstance(reason, str) or not 20 <= len(reason) <= 350 or any(ord(c) < 32 for c in reason):
        raise ValueError("Invalid explanation")
    references = re.findall(r"\[(d[1-9][0-9]*)\]", reason)
    by_id = {item["id"]: item for item in evidence}
    if not references or not set(references) <= by_id.keys() or (target and target not in references):
        raise ValueError("Explanation must cite the supplied target evidence")
    prose = re.sub(r"\[d[1-9][0-9]*\]", "", reason)
    if re.search(r"[\[\]{}<>]|https?://|\d|%", prose):
        raise ValueError("Explanation must use references, not new numeric claims or markup")
    allowed = {by_id[ident]["label"].casefold() for ident in references}
    for label in sorted(allowed, key=len, reverse=True):
        prose = re.sub(r"(?<!\w)" + re.escape(label) + r"(?:s|es)?(?!\w)", "", prose, flags=re.I)
    # Reject uncited names and common unsupported state/background assertions.
    # This remains a limited lexical check, not semantic proof of arbitrary prose.
    prohibited = (set(known_object_labels()) - allowed) | {
        "people", "pedestrian", "vehicle", "sofa", "television", "building", "balcony",
        "road", "pavement", "tree", "window", "wall", "safe", "safely", "unsafe",
        "danger", "dangerous", "hazard", "distance", "near", "far", "moving", "stationary",
        "crossing", "collision", "emergency", "urgent", "red", "green", "blue"}
    if "person" in allowed:
        prohibited -= {"people", "pedestrian"}
    # Conditional advice can mention moving; remove just this bounded condition.
    prose = re.sub(r"if moving nearby", "", prose, flags=re.I)
    for label in prohibited:
        if re.search(r"(?<!\w)" + re.escape(label) + r"(?:s|es)?(?!\w)", prose, flags=re.I):
            raise ValueError("Explanation must use references and avoid unsupported scene claims")


def resolve_reason(reason, evidence):
    by_id = {item["id"]: item for item in evidence}
    def replacement(match):
        item = by_id[match[1]]
        named = re.search(r"(?<!\w)" + re.escape(item["label"]) + r"(?:s|es)?(?!\w)", reason, flags=re.I)
        return (f"[{item['id']}, {item['confidence_percent']:g}% YOLO]" if named else
                f"{item['label']} [{item['id']}] ({item['confidence_percent']:g}% YOLO)")
    return re.sub(r"\[(d[1-9][0-9]*)\]", replacement, reason)


def render_description(evidence, total, width, height, cfg, choice=None, status="validated",
                       context_override=None, reason_note=""):
    """Render detector-owned facts/actions and a checked, attributed explanation."""
    choices = action_choices(evidence)
    environment = choice["environment"] if choice else "unclear"
    setting = choice["setting"] if choice else "unclear"
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
        context = context_override or environment_text(environment, setting)
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
    if reason_note:
        reason += " " + reason_note
    generated_reason = choice.get("reason") if choice and status == "validated" else None
    if generated_reason:
        reason = "LLM explanation: " + resolve_reason(generated_reason, evidence)
    scores = "; ".join(
        f"{item['label']} ({item.get('position', 'classification')}, {item['confidence_percent']:g}%)"
        for item in evidence)
    summary = "YOLO scores: " + (scores or "No predictions above the selected threshold.")
    detections = facts + (f"\n{summary}" if evidence and not classification else "")
    if cfg.describe_backend in NO_ENVIRONMENT_BACKENDS:
        text = f"Detections: {detections}\n\nDecision: {decision}\n{advice}\n\nReason: {reason}"
        environment = setting = None
    elif cfg.describe_backend == "LLM":
        text = f"Detections: {detections}\n\nEnvironment (LLM): {context}\n\nDecision (LLM): {decision}\n{advice}\n\nDescription (LLM): {reason}"
    else:
        text = f"Detections: {detections}\n\nEnvironment: {context}\n\nDecision: {decision}\n{advice}\n\nReason: {reason}"
    return {
        "text": text,
        "grounding_summary": summary,
        "model": cfg.describe_model,
        "evidence": evidence, "total_predictions": total, "frame_size": [width, height],
        "grounding_status": status, "environment": environment, "setting": setting,
        "backend": cfg.describe_backend, **method_details(cfg),
        "action_decision": decision,
        "reason": reason,
        "reason_source": "LLM" if generated_reason else "application / algorithm",
        **({"model_explanation": generated_reason} if generated_reason else {}),
        "suggested_action": {**step, "text": advice},
    }
