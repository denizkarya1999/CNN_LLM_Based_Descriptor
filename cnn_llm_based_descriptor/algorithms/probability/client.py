"""Probability-based descriptor: infer a scene and select one review suggestion."""
from ...methods import method_details
from ..shared.evidence import detection_evidence
from ..shared.grounding import render_description
from ..shared.knowledge import load_scene_model
from ..shared.scenes import SETTINGS, environment_text
from ..shared.review_policy import assess_detections
from .inference import infer_setting, infer_review
from .explanation import probability_explanation


class ProbabilityClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = load_scene_model()

    def describe(self, frame, records):
        height, width = frame.shape[:2]
        classification = self.cfg.task == "Image classification"
        # Only LLM prompts have an object budget. Classical methods use all
        # predictions returned by YOLO after its confidence/class/NMS filters.
        evidence, total = detection_evidence(records, width, height, None,
                                            self.cfg.confidence, require_boxes=not classification)
        details = method_details(self.cfg)
        if classification or not evidence:
            result = render_description(evidence, total, width, height, self.cfg,
                                        status="classification" if classification else "no_detections")
            operation = "Probability calculation"
            result["text"] += f"\n\nDescription ({self.cfg.describe_backend}): {operation} was not performed because no eligible object-box evidence was available."
        else:
            scene = infer_setting(evidence, self.model)
            decision = infer_review(assess_detections(evidence, self.model), self.model)
            step, source = decision["next_step"], "probability model"
            note = (f"Conditional review probabilities were evaluated for all {len(evidence)} detections. "
                    f"Highest review probability: {decision['selected_probability']*100:.0f}%; "
                    f"recommendation threshold: {decision['threshold']*100:.0f}%. "
                    "These are illustrative policy probabilities, not measured accuracy or danger.")
            extra = {"scene_inference": scene, "decision_inference": decision}
            setting = scene["setting"]
            environment = SETTINGS[setting][1]
            context = environment_text(environment, setting, source)
            ranking = "; ".join(f"{SETTINGS[name][0]} {scene['posterior'][name]*100:.1f}%"
                                for name in scene["candidates"] if scene["posterior"][name] >= .01)
            context += f"\nPossible settings: {ranking}.\nIllustrative model probabilities, not measured accuracy."
            if scene["custom_labels"]:
                context += " Custom labels without scene associations use a neutral fallback."
            choice = {"environment": environment, "setting": setting, "next_step": step}
            result = render_description(evidence, total, width, height, self.cfg, choice=choice,
                                        context_override=context, reason_note=note)
            result.update(extra)
            reasoning = probability_explanation(scene, decision)
            result["reasoning_trace"] = reasoning
            result["text"] += "\n\n" + reasoning
            result["efficiency"] = {"scene_hypotheses": len(scene["posterior"]),
                                    "likelihood_factors": sum(map(len, scene["likelihood_factors"].values())),
                                    "review_assessments": len(decision["assessments"])}
        result.update(model=details["algorithm"], backend=self.cfg.describe_backend, **details,
                      detections_considered=len(evidence), knowledge_base_labels=len(self.model["labels"]))
        return result
