"""Deterministic robot recommendations from every eligible YOLO box.

Rules are an explicit demonstration policy, not navigation or a learned model.
Counts concern one filtered frame; no physical movement is executed here.
"""
from collections import Counter
from dataclasses import dataclass
import operator

from .grounding import LOW_SCORE, render_description
from .methods import GENERAL_ALGORITHM, method_details

GROUPS = {
    'people': frozenset({'person'}),
    'vehicles': frozenset({'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'boat', 'airplane'}),
    'signals': frozenset({'traffic light', 'stop sign'}),
    'animals': frozenset({'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'}),
    'tools': frozenset({'knife', 'scissors'}),
}
OPERATORS = {'>=': operator.ge, '<=': operator.le, '==': operator.eq}


@dataclass(frozen=True)
class RobotPlan:
    key: str
    name: str
    conditions: tuple
    groups: tuple
    take_action: bool
    steps: tuple


# Earlier matched action rules take priority. All matching action steps survive.
PLANS = (
    RobotPlan('crowded_area', 'Crowded-Area Robot Action Plan', (('people', '>=', 5),), ('people',), True,
              ('Pause any intended movement near the detected people.',
               'Review their current positions and allow space before resuming a separately verified route.')),
    RobotPlan('people_and_vehicles', 'People-and-Vehicles Robot Action Plan',
              (('people', '>=', 1), ('vehicles', '>=', 1)), ('people', 'vehicles'), True,
              ('Hold intended movement while checking the detected people and vehicles.',
               'Recheck their positions before choosing any movement.')),
    RobotPlan('signal_review', 'Signal-Review Robot Action Plan', (('signals', '>=', 1),), ('signals',), True,
              ('Check the actual state of each detected signal before any related movement.',)),
    RobotPlan('animal_presence', 'Animal-Presence Robot Action Plan', (('animals', '>=', 1),), ('animals',), True,
              ('If moving nearby, pause and review the detected animals\' current positions before proceeding.',)),
    RobotPlan('tool_handling', 'Tool-Handling Robot Action Plan', (('tools', '>=', 1),), ('tools',), True,
              ('Before any planned handling, inspect the detected tool and use appropriate handling precautions.',)),
    RobotPlan('uncertain_detections', 'Uncertain-Detection Robot Action Plan', (('tentative', '>=', 1),), ('tentative',), True,
              ('Acquire another view to verify the below-50% detections before using those labels for an action.',)),
    RobotPlan('dense_objects', 'Dense-Object Robot Action Plan', (('strong', '>=', 10),), ('strong',), True,
              ('Pause intended movement and review the positions of the detected objects.',
               'Reassess the view before choosing any movement; object count does not identify a clear route.')),
    RobotPlan('vehicle_presence', 'Vehicle-Presence Robot Action Plan', (('vehicles', '>=', 1),), ('vehicles',), True,
              ('If moving nearby, hold intended movement and inspect the detected vehicles\' current positions.',)),
    RobotPlan('people_nearby', 'People-Presence Robot Action Plan',
              (('people', '>=', 1), ('people', '<=', 4)), ('people',), True,
              ('If moving nearby, allow space around the detected people and recheck their positions.',)),
    RobotPlan('empty_area', 'Empty-Area Robot Action Plan', (('total', '==', 0),), (), False,
              ('No new robot action is requested by this detection-based policy; retain the current idle/monitoring state.',
               'Keep sampling frames. Zero eligible boxes does not establish that the physical area is empty or clear.')),
    RobotPlan('sparse_objects', 'Sparse-Object Robot Action Plan',
              (('strong', '>=', 1), ('strong', '<=', 3), ('tentative', '==', 0)), ('strong',), False,
              ('Keep the current idle/monitoring state; the small object count alone requests no intervention.',)),
    RobotPlan('moderate_objects', 'Moderate-Object Robot Action Plan',
              (('strong', '>=', 4), ('strong', '<=', 9), ('tentative', '==', 0)), ('strong',), False,
              ('Keep the current idle/monitoring state; this object count alone requests no intervention.',)),
)


def evaluate_plans(evidence):
    """Evaluate every rule, keep all matches, and let any action override idle."""
    members = {name: [] for name in ('total', 'strong', 'tentative', *GROUPS)}
    assessments = []
    for item in evidence:
        groups = ['total', 'strong' if item['confidence'] >= LOW_SCORE else 'tentative']
        if item['confidence'] >= LOW_SCORE:
            groups.extend(name for name, labels in GROUPS.items() if item['label'].casefold() in labels)
        for group in groups:
            members[group].append(item['id'])
        assessments.append({'detection_id': item['id'], 'label': item['label'],
                            'confidence': item['confidence'], 'groups': groups})
    counts = {name: len(ids) for name, ids in members.items()}
    evaluations, matches = [], []
    for priority, plan in enumerate(PLANS, 1):
        checks = [{'metric': metric, 'value': counts[metric], 'operator': op, 'threshold': threshold,
                   'passed': OPERATORS[op](counts[metric], threshold)} for metric, op, threshold in plan.conditions]
        matched = all(check['passed'] for check in checks)
        entry = {'id': plan.key, 'name': plan.name, 'priority': priority,
                 'action_decision': 'Take Action' if plan.take_action else 'No Action',
                 'conditions': checks, 'matched': matched}
        evaluations.append(entry)
        if matched:
            ids = set(ident for group in plan.groups for ident in members[group])
            matches.append({**entry, 'detection_ids': [item['id'] for item in evidence if item['id'] in ids],
                            'steps': list(plan.steps)})
    active = [plan for plan in matches if plan['action_decision'] == 'Take Action']
    selected = (active or matches)[0]
    applicable = active or [selected]
    match_ids = {plan['id']: set(plan['detection_ids']) for plan in matches}
    for item in assessments:
        item['matched_plan_ids'] = [plan['id'] for plan in matches
                                   if item['detection_id'] in match_ids[plan['id']]]
    return {'policy_version': 1, 'selected_plan': selected, 'matched_plans': matches,
            'applicable_plans': applicable, 'evaluations': evaluations, 'counts': counts,
            'assessments': assessments, 'confidence_cutoff': LOW_SCORE,
            'rules_evaluated': len(evaluations), 'conditions_evaluated': sum(len(p.conditions) for p in PLANS),
            'selection_policy': 'First matched action plan in priority order; retain every matched action plan. Idle plans apply only if no action rule matches.'}


def condition_text(plan):
    return ' AND '.join(f"{c['metric']}={c['value']} {c['operator']} {c['threshold']} "
                        f"({'pass' if c['passed'] else 'fail'})" for c in plan['conditions'])


class GeneralAlgorithmicClient:
    def __init__(self, cfg):
        self.cfg = cfg

    def describe(self, frame, records):
        from .descriptor import detection_evidence
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
