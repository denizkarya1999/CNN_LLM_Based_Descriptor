"""Normalize and filter YOLO evidence identically for every approach."""
import math


def detection_evidence(records, width, height, limit, threshold=0.0, require_boxes=False):
    """Keep exact scores and spatial evidence. limit=None retains every valid prediction."""
    candidates = []
    for record in records:
        try:
            confidence = float(record["confidence"])
            if not math.isfinite(confidence) or not threshold <= confidence <= 1:
                continue
            item = {"label": " ".join(str(record.get("name", "object")).split())[:100],
                    "confidence": confidence, "confidence_percent": round(confidence * 100, 2)}
            box = record.get("box")
            if require_boxes and not box:
                continue
            if box:
                xs = [float(box[key]) for key in ("x1", "x2", "x3", "x4") if key in box]
                ys = [float(box[key]) for key in ("y1", "y2", "y3", "y4") if key in box]
                if len(xs) < 2 or len(xs) != len(ys) or not all(map(math.isfinite, xs + ys)):
                    continue
                x1, x2 = [max(0, min(width, x)) for x in (min(xs), max(xs))]
                y1, y2 = [max(0, min(height, y)) for y in (min(ys), max(ys))]
                if x2 <= x1 or y2 <= y1:
                    continue
                item["box_xyxy_pixels"] = [x1, y1, x2, y2]
                item["box_xyxy_normalized"] = [round(x1/width, 4), round(y1/height, 4),
                                                round(x2/width, 4), round(y2/height, 4)]
                cx, cy = (x1+x2)/(2*width), (y1+y2)/(2*height)
                item["position"] = ("top" if cy < 1/3 else "bottom" if cy > 2/3 else "middle") + "-" + (
                    "left" if cx < 1/3 else "right" if cx > 2/3 else "center")
            candidates.append(item)
        except (KeyError, TypeError, ValueError):
            continue
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    for index, item in enumerate(candidates, 1):
        item["id"] = f"d{index}"
    return candidates[:limit], len(candidates)
