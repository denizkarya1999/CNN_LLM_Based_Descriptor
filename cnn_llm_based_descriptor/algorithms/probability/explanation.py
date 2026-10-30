"""Render scene posterior arithmetic and conditional review probabilities."""
from ..shared.action_names import ACTION_NAMES
from ..shared.scenes import SETTINGS


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
