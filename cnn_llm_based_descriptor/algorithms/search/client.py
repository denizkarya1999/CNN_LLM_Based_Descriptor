"""Search-based descriptor: build and explain a complete object-review plan."""
from ...methods import method_details
from ..shared.evidence import detection_evidence
from ..shared.grounding import render_description
from ..shared.knowledge import load_scene_model
from ..shared.scenes import SETTINGS, environment_text
from .targets import search_review_plan
from .explanation import search_explanation


class SearchClient:
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
            operation = "Search traversal"
            result["text"] += f"\n\nDescription ({self.cfg.describe_backend}): {operation} was not performed because no eligible object-box evidence was available."
        else:
            scene = {"setting": "unclear", "custom_labels": []}
            plan = search_review_plan(evidence, self.cfg.describe_search, self.model)
            step = plan["steps"][0]["choice"] if plan["steps"] else "no_action"
            source = "knowledge-base"
            note = (f"{self.cfg.describe_search} assessed all {len(evidence)} detections and planned "
                    f"{len(plan['steps'])} review steps in {len(plan['batches'])} batch(es).")
            if plan["steps"]:
                note += " The first suggestion is shown as the decision."
            if not plan["steps"]:
                note += " No review rule was triggered by the detected labels and scores."
            extra = {"decision_search": plan}
            setting = scene["setting"]
            environment = SETTINGS[setting][1]
            context = environment_text(environment, setting, source)
            choice = {"environment": environment, "setting": setting, "next_step": step}
            result = render_description(evidence, total, width, height, self.cfg, choice=choice,
                                        context_override=context, reason_note=note)
            result.update(extra)
            reasoning = search_explanation(plan, evidence)
            result["reasoning_trace"] = reasoning
            result["text"] += "\n\n" + reasoning
            result["efficiency"] = {"expanded_states": plan["expanded_states"],
                                    "generated_states": plan["generated_states"],
                                    "peak_frontier_nodes": plan["peak_frontier"],
                                    "planned_reviews": len(plan["steps"])}
        result.update(model=details["algorithm"], backend=self.cfg.describe_backend, **details,
                      detections_considered=len(evidence), knowledge_base_labels=len(self.model["labels"]))
        return result
