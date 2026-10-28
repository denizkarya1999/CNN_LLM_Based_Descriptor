"""One timing boundary shared by live descriptions and same-frame comparisons."""
import time


def efficiency_text(result):
    data = result.get('efficiency', {})
    if 'expanded_states' in data:
        return (f"{data['expanded_states']} expanded / {data['generated_states']} generated states; "
                f"peak frontier {data['peak_frontier_nodes']} nodes")
    if 'likelihood_factors' in data:
        return (f"{data['scene_hypotheses']} hypotheses × "
                f"{data['likelihood_factors']//data['scene_hypotheses']} observed classes; "
                f"{data['review_assessments']} review assessments")
    if 'eval_count' in data:
        return f"{data.get('prompt_eval_count', '—')} input tokens; {data['eval_count']} output tokens"
    return 'No inference work reported'


def measure_description(client, frame, records):
    started = time.perf_counter_ns()
    result = client.describe(frame, records)
    elapsed_ns = time.perf_counter_ns() - started
    result.update(inference_latency_ms=elapsed_ns/1_000_000, elapsed_seconds=elapsed_ns/1_000_000_000)
    result['efficiency_summary'] = efficiency_text(result)
    result['text'] += '\n\nEfficiency: ' + result['efficiency_summary']
    result['measurement_scope'] = 'Descriptor call including preparation, inference and explanation formatting; excludes YOLO, queue wait, GUI and saving. Model loading is included when needed.'
    return result
