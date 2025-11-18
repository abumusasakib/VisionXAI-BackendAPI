#!/usr/bin/env python3
"""Decode attention image bytes from tools/response.json and validate schema.

Writes:
- tools/attention_decoded.png (decoded PNG)
- tools/decode_report.json (validation + paths)

Run: python .\tools\decode_attention_image.py
"""
import base64
import io
import json
import os
import sys
from typing import Any, Dict, List

try:
    from PIL import Image
except Exception:
    Image = None


ROOT = os.path.dirname(__file__)
RESP_PATH = os.path.join(ROOT, "response.json")
OUT_PNG = os.path.join(ROOT, "attention_decoded.png")
REPORT = os.path.join(ROOT, "decode_report.json")


def fail(msg: str, extra: Dict[str, Any] = None):
    report = {"success": False, "messages": [msg]}
    if extra:
        report.update(extra)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(msg)
    sys.exit(1)


def load_response(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        fail(f"Response file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def decode_base64_to_png(b64: str, out_path: str) -> None:
    # Strip data URL prefix if present
    if b64.startswith("data:"):
        try:
            b64 = b64.split(",", 1)[1]
        except Exception:
            pass
    raw = base64.b64decode(b64)
    # Try using PIL if available, else write raw bytes
    if Image:
        try:
            img = Image.open(io.BytesIO(raw))
            img.save(out_path)
            return
        except Exception:
            # fallback: write bytes
            pass
    with open(out_path, "wb") as f:
        f.write(raw)


def validate_topk(items: Any) -> List[str]:
    msgs: List[str] = []
    if not isinstance(items, list):
        msgs.append("attention_topk_items is not a list")
        return msgs
    for i, row in enumerate(items):
        if not isinstance(row, list):
            msgs.append(f"topk row {i} is not a list")
            continue
        for j, it in enumerate(row):
            if not isinstance(it, dict):
                msgs.append(f"topk item [{i}][{j}] is not a dict")
                continue
            for k in ("row", "col", "score"):
                if k not in it:
                    msgs.append(f"topk item [{i}][{j}] missing key '{k}'")
            # type checks
            if "row" in it and not isinstance(it["row"], (int, float)):
                msgs.append(f"topk item [{i}][{j}] 'row' not numeric")
            if "col" in it and not isinstance(it["col"], (int, float)):
                msgs.append(f"topk item [{i}][{j}] 'col' not numeric")
            if "score" in it and not isinstance(it["score"], (int, float)):
                msgs.append(f"topk item [{i}][{j}] 'score' not numeric")
    return msgs


def main():
    resp = load_response(RESP_PATH)
    messages: List[str] = []
    # Decode image
    b64 = resp.get("attention_image_bytes") or resp.get("attention_image")
    if not b64:
        fail("No 'attention_image_bytes' or 'attention_image' field found in response.json")
    try:
        decode_base64_to_png(b64, OUT_PNG)
        messages.append(f"Decoded image written to {OUT_PNG}")
    except Exception as e:
        fail(f"Failed decoding image: {e}")

    # Validate topk items
    topk = resp.get("attention_topk_items") or resp.get("attention_topk")
    if topk is None:
        messages.append("No 'attention_topk_items' or 'attention_topk' field present")
        topk_msgs = ["missing"]
    else:
        topk_msgs = validate_topk(topk)
        if not topk_msgs:
            messages.append("attention_topk_items schema OK")

    # Validate attention_grid
    grid = resp.get("attention_grid")
    if grid is None:
        messages.append("No 'attention_grid' field present")
        grid_ok = False
    else:
        grid_ok = isinstance(grid, list) and len(grid) == 2 and all(isinstance(x, int) for x in grid)
        messages.append(f"attention_grid present: {grid}; valid={grid_ok}")

    # Validate attention_shape
    shape = resp.get("attention_shape")
    if shape is None:
        messages.append("No 'attention_shape' field present")
        shape_ok = False
    else:
        shape_ok = isinstance(shape, dict) and "rows" in shape and "cols" in shape and isinstance(shape["rows"], int) and isinstance(shape["cols"], int)
        messages.append(f"attention_shape present: {shape}; valid={shape_ok}")

    report = {
        "success": True,
        "messages": messages,
        "topk_validation_messages": topk_msgs,
        "attention_grid": grid,
        "attention_grid_valid": grid_ok,
        "attention_shape": shape,
        "attention_shape_valid": shape_ok,
        "decoded_image": OUT_PNG,
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Decoder completed. Report written to:", REPORT)
    for m in messages:
        print("-", m)


if __name__ == "__main__":
    main()
