"""Human-readable audit output from the conventional algorithms' actual results."""
from .scenes import SETTINGS

ACTION_NAMES = {"verify": "verify label", "inspect": "inspect", "check_signal": "check state",
                "handle_carefully": "care if handled", "no_action": "presence only"}


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


def probability_explanation(scene, decision):
    lines = ['Description (Probability Algorithm):', 'Label probabilities:', 'YOLO scores are detector confidence; P(review useful) is the separate, illustrative policy table.']
    for item in decision['assessments']:
        lines.append(f"• {item['label']} [{item['detection_id']}] · YOLO {item['confidence']:.1%} · P(review useful | label, score band) = {item['review_probability']:.1%} → {ACTION_NAMES[item['kind']]}")
    lines.extend(['', 'Scenario probabilities:', 'Inference by Enumeration: posterior ∝ prior × product of observed-label likelihoods.'])
    for name in sorted(scene['posterior'], key=scene['posterior'].get, reverse=True):
        lines.append(f"• {SETTINGS[name][0]}: prior {scene['priors'][name]:.5f}; log joint {scene['log_joint'][name]:.5f}; exp(log joint − {scene['normalization_offset']:.5f}) / {scene['scaled_normalizer']:.5f} = {scene['posterior'][name]:.2%}")
    top = max(scene['posterior'], key=scene['posterior'].get)
    lines.extend(['', f"Calculation for leading scenario: {SETTINGS[top][0]}"])
    factors = ' × '.join(f'{v:.5f}' for v in scene['likelihood_factors'][top].values())
    lines.append(f"Unnormalized joint = {scene['priors'][top]:.5f} × ({factors}).")
    lines.append('The implementation adds log likelihoods before normalization to avoid numerical underflow.')
    for label, score in scene['label_confidences'].items():
        lines.append(f"• {label} · YOLO {score:.1%} · likelihood = {scene['evidence_weights'][label]:.2f} × {scene['raw_likelihoods'][top][label]:.5f} + {1-scene['evidence_weights'][label]:.2f} × {scene['predictive_means'][label]:.5f} = {scene['likelihood_factors'][top][label]:.5f}")
    lines.append('Duplicate classes contribute once at their highest score. Low-score likelihoods are weakened; missing classes are unobserved; custom labels are neutral.')
    lines.append('Decision calculation: max(' + ', '.join(f"{item['review_probability']:.2f}" for item in decision['assessments']) + f") = {decision['selected_probability']:.2f}; compare with {decision['threshold']:.2f} → {'Take Action' if decision['next_step'] != 'no_action' else 'No Action'}.")
    lines.append(f"Decision threshold: {decision['threshold']:.0%}; largest review probability: {decision['selected_probability']:.1%}.")
    lines.append('A specific place needs two strong informative cues, posterior ≥55%, and a ≥15-point lead. Otherwise the setting remains unclear.')
    lines.append('These are hand-authored model probabilities, not measured accuracy.')
    return '\n'.join(lines)
