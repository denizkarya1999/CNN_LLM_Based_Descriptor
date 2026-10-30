"""Deterministic robot recommendations from every eligible YOLO box.

Rules are an explicit demonstration policy, not navigation or a learned model.
Counts concern one filtered frame; no physical movement is executed here.
"""
from collections import Counter
from dataclasses import dataclass
import operator

from ..shared.grounding import LOW_SCORE

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
