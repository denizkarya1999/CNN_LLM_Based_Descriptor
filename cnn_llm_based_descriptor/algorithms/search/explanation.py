"""Render the actual search traversal, chosen plan and work counters."""
from ..shared.action_names import ACTION_NAMES


def search_explanation(plan, evidence):
    by_id = {item['id']: item for item in evidence}

    def step_name(choice):
        action, ident = choice.split(':', 1)
        return f"{by_id[ident]['label']} [{ident}] / {ACTION_NAMES.get(action, action)}"

    lines = ['Description (Search Algorithm):', 'Search traversal:',
             f"{plan['algorithm']} visits review-plan states built from detected labels, their confidence bands, and allowed suggestions."]
    if not plan['traversal']:
        lines.append('Start → No Action: no detected label/score triggered a review rule; no frontier was expanded.')
    for entry in plan['traversal']:
        route = 'Start' + ''.join(' → ' + step_name(choice) for choice in entry['path'])
        limit = f" · depth limit {entry['depth_limit']}" if entry['depth_limit'] is not None else ''
        priority = f" · priority {entry['priority']:.3f}" if entry['priority'] is not None else ''
        lines.append(f"{entry['batch']}.{entry['visit']}{limit}: {route} · g={entry['g']:.3f}, h={entry['h']:.3f}{priority}" + (' · GOAL' if entry['goal'] else ''))
    if plan['trace_omitted']:
        lines.append(f"Trace display/store limit reached: {plan['trace_omitted']} later expansions omitted; all targets were still processed.")
    lines.extend(['', 'Selected review path:'])
    lines.append(' → '.join(step_name(s['choice']) for s in plan['steps']) or 'No Action')
    lines.append(f"Plan cost {plan['cost']:.3f}; {plan['expanded_states']} expanded states; {plan['generated_states']} generated states; peak frontier {plan['peak_frontier']} nodes.")
    lines.append('Costs are image-review effort and attention shifts. Each batch has up to five targets; UCS/A* minimize cost within a batch.')
    lines.extend(['', 'Detection decision associations:'])
    for item in plan['assessments']:
        lines.append(f"• {item['label']} [{item['detection_id']}] · YOLO {item['confidence']:.1%} → {ACTION_NAMES[item['kind']]}")
    return '\n'.join(lines)
