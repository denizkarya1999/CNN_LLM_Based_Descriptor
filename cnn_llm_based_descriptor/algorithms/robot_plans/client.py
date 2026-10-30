"""Render named robot action plans and their complete rule audit."""
from collections import Counter
from ...methods import GENERAL_ALGORITHM, method_details
from ..shared.evidence import detection_evidence
from ..shared.grounding import render_description
from .plans import GROUPS, evaluate_plans, condition_text


class GeneralAlgorithmicClient:
    def __init__(self, cfg):
        self.cfg = cfg

    def describe(self, frame, records):
        height, width = frame.shape[:2]
        classification = self.cfg.task == 'Image classification'
        evidence, total = detection_evidence(records, width, height, None, self.cfg.confidence,
                                            require_boxes=not classification)
        result = render_description(evidence, total, width, height, self.cfg,
                                    status='classification' if classification else 'validated')
        result.update(model=GENERAL_ALGORITHM, **method_details(self.cfg), detections_considered=len(evidence))
        if classification:
            result['text'] += '\n\nDescription (General Algorithmic Approach): Robot plans require object boxes; classification cannot establish object counts.'
            result['efficiency'] = {'rules_evaluated': 0, 'conditions_evaluated': 0, 'detections_assessed': 0}
            return result
        plan = evaluate_plans(evidence)
        selected, counts = plan['selected_plan'], plan['counts']
        reason = f"{selected['name']} matched: {condition_text(selected)}. "
        if selected['action_decision'] == 'Take Action':
            reason += 'It is the highest-priority action match; all other matched action steps are retained.'
        else:
            reason += 'No action rule matched.'
        if not evidence:
            reason += f' Zero eligible YOLO boxes at or above {self.cfg.confidence*100:g}% is a detector result, not confirmation of an empty area.'
        steps = []
        for matched in plan['applicable_plans']:
            steps.append(matched['name'] + ':')
            steps.extend(f'  {index}. {step}' for index, step in enumerate(matched['steps'], 1))
        steps_text = '\n'.join(steps)
        lines = ['Description (General Algorithmic Approach):',
                 f"Counts: {counts['total']} eligible boxes; {counts['strong']} at least 50%; {counts['tentative']} tentative.",
                 'Group counts (at least 50% confidence): ' + '; '.join(f'{group}={counts[group]}' for group in GROUPS) + '.',
                 'Counts are per-frame detections, not unique tracked objects or physical density. All labels contribute to total counts.',
                 'Rule evaluation (priority order):']
        for rule in plan['evaluations']:
            state = 'MATCH' if rule['matched'] else 'no match'
            disposition = ('; selected' if rule['id'] == selected['id'] else
                           '; idle plan overridden' if rule['matched'] and rule['action_decision'] == 'No Action' else '')
            lines.append(f"{rule['priority']}. {rule['name']} — {condition_text(rule)} → {state}{disposition}")
        lines.append('Detection assignments:')
        for item in plan['assessments']:
            lines.append(f"[{item['detection_id']}] {item['label']} ({item['confidence']*100:g}% YOLO) → "
                         + ', '.join(item['groups']))
        lines.extend(['Hand-authored count thresholds are a demonstration policy, not learned accuracy.',
                      'Robot steps are recommendations only; this application does not send robot commands.'])
        reasoning = '\n'.join(lines)
        facts = result['text'].partition('\n\nDecision:')[0]
        label_counts = dict(Counter(item['label'] for item in evidence))
        result.update(text=(f"{facts}\n\nAction Plan: {selected['name']}\n\nDecision: {selected['action_decision']}"
                            f"\n\nReason: {reason}\n\nRobot steps:\n{steps_text}\n\n{reasoning}"),
                      action_decision=selected['action_decision'], reason=reason,
                      suggested_action={'kind': selected['id'], 'detection_id': None,
                                        'detection_ids': selected['detection_ids'], 'text': steps_text},
                      action_plan=plan, action_plan_name=selected['name'], label_counts=label_counts, reasoning_trace=reasoning,
                      efficiency={'rules_evaluated': plan['rules_evaluated'],
                                  'conditions_evaluated': plan['conditions_evaluated'],
                                  'detections_assessed': len(evidence), 'matched_plans': len(plan['matched_plans'])})
        return result
