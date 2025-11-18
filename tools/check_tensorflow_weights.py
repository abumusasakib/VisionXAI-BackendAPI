import os
import sys
import argparse
import hashlib
from pathlib import Path
import datetime
import json


def compute_sha256(file_path):
    """Compute SHA256 using a pure-Python implementation (cross-platform)."""
    try:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest().upper()
    except Exception as e:
        print(f"⚠️ Error computing checksum for {file_path}: {e}")
        return None

def find_checkpoints(weights_dir: Path):
    """Find available checkpoint prefixes under the given directory.

    Returns a dict mapping parent_dir -> list of checkpoint_prefixes (without extensions).
    """
    checkpoints = {}
    for root, dirs, files in os.walk(weights_dir):
        root_path = Path(root)
        # find files matching *.data-00000-of-00001
        data_files = [f for f in files if f.endswith(".data-00000-of-00001")]
        if not data_files:
            continue
        prefixes = [f.replace(".data-00000-of-00001", "") for f in data_files]
        checkpoints[str(root_path)] = prefixes
    return checkpoints


def inspect_tokenizer(tokenizer_path: Path, head_bytes: int = 4096, unpickle: bool = False):
    """Peek into a tokenizer pickle file to guess its origin (safe, no unpickle by default).

    If `unpickle` is True this will attempt to unpickle the object and extract
    useful metadata (vocab size, sample tokens). Unpickling can execute code
    in the pickle; only enable when you trust the environment.

    Returns a dict with fields:
      - exists: bool
      - head: printable preview of the first bytes
      - found: list of detected keywords in the head
      - unpickle_info: optional dict with unpickled metadata (if unpickle=True)
      - error: optional error string
    """
    out = {"exists": False, "head": None, "found": [], "unpickle_info": None}
    try:
        if not tokenizer_path.exists():
            return out
        out["exists"] = True
        with open(tokenizer_path, "rb") as f:
            data = f.read(head_bytes)
        # Produce a sanitized head string with printable ASCII and dots
        s = ''.join([chr(b) if 32 <= b < 127 else '.' for b in data[: min(len(data), head_bytes)]])
        out["head"] = s
        keywords = ["Tokenizer", "keras", "keras_preprocessing", "word_index", "index_word", "TextVectorization"]
        found = [kw for kw in keywords if kw in s]
        out["found"] = found

        if unpickle:
            try:
                import pickle

                # Try to unpickle; warn user in logs that this may execute code.
                with open(tokenizer_path, "rb") as f:
                    obj = pickle.load(f)

                info = {"type": type(obj).__name__}

                # Keras Tokenizer
                if hasattr(obj, "word_index"):
                    try:
                        wi = getattr(obj, "word_index")
                        info["word_index_len"] = len(wi)
                        # sample first 20 items
                        sample = list(wi.items())[:20]
                        info["word_index_sample"] = sample
                    except Exception as e:
                        info["word_index_error"] = str(e)

                # Keras Tokenizer: index_word
                if hasattr(obj, "index_word"):
                    try:
                        iw = getattr(obj, "index_word")
                        info["index_word_len"] = len(iw)
                        sample = list(iw.items())[:20]
                        info["index_word_sample"] = sample
                    except Exception as e:
                        info["index_word_error"] = str(e)

                # If object is a plain list (vocab)
                if isinstance(obj, (list, tuple)):
                    info["vocab_len"] = len(obj)
                    info["vocab_sample"] = list(obj)[:20]

                # If object is a dict
                if isinstance(obj, dict):
                    info["dict_len"] = len(obj)
                    try:
                        # try to show first 20 key->value pairs
                        items = list(obj.items())[:20]
                        info["dict_sample"] = items
                    except Exception as e:
                        info["dict_sample_error"] = str(e)

                out["unpickle_info"] = info
            except Exception as e:
                out["unpickle_info"] = {"error": f"Unpickle failed: {e}"}

        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def check_tensorflow_weights(directory, expected_sha256=None, try_load=False, show_all=False, unpickle_tokenizer=False):
    weights_dir = Path(directory)
    if not weights_dir.exists():
        print(f"❌ Directory not found: {weights_dir}")
        return None

    report_lines = []
    report_lines.append(f"Report generated: {datetime.datetime.utcnow().isoformat()} UTC")
    report_lines.append(f"Scanning for TensorFlow checkpoints under: {weights_dir}")

    # Inspect tokenizer if present (optionally unpickle)
    tokenizer_pkl = weights_dir / "tokenizer.pkl"
    tok_info = inspect_tokenizer(tokenizer_pkl, unpickle=unpickle_tokenizer)
    report_lines.append("\n== Tokenizer Inspection ==")
    if not tok_info.get("exists"):
        report_lines.append(f"Tokenizer not found at: {tokenizer_pkl}")
    else:
        head_preview = tok_info.get("head")
        short_head = (head_preview[:400] + "...") if head_preview and len(head_preview) > 400 else head_preview
        report_lines.append(f"Tokenizer path: {tokenizer_pkl}")
        report_lines.append(f"Head (preview): {short_head}")
        if tok_info.get("found"):
            report_lines.append(f"Detected keywords in tokenizer file: {tok_info.get('found')}")
        if tok_info.get("error"):
            report_lines.append(f"Tokenizer inspect error: {tok_info.get('error')}")
        # If detailed unpickle info is available, include it in the report
        if tok_info.get("unpickle_info") is not None:
            try:
                ui = tok_info.get("unpickle_info")
                # pretty print as JSON for readability
                report_lines.append("Tokenizer unpickle_info:")
                report_lines.append(json.dumps(ui, indent=2, ensure_ascii=False, default=str))
            except Exception as e:
                report_lines.append(f"Failed to render unpickle_info: {e}")

    ckpts = find_checkpoints(weights_dir)
    if not ckpts:
        report_lines.append("No checkpoints found under the given directory.")
        print("❌ No checkpoints found under the given directory.")
        # still return the report content
        return "\n".join(report_lines)

    for root, prefixes in ckpts.items():
        report_lines.append(f"\nDirectory: {root}")
        for prefix in prefixes:
            data_file = Path(root) / f"{prefix}.data-00000-of-00001"
            index_file = Path(root) / f"{prefix}.index"
            report_lines.append(f"\n- Checkpoint prefix: {prefix}")
            report_lines.append(f"  data:  {data_file}")
            report_lines.append(f"  index: {index_file}")

            if not data_file.exists() or not index_file.exists():
                report_lines.append("  Missing data or index file for this checkpoint")
                continue

            # Compute SHA256
            sha = compute_sha256(data_file)
            if sha:
                report_lines.append(f"  SHA256: {sha}")
                if expected_sha256 and sha.upper() == expected_sha256.upper():
                    report_lines.append("  ✅ Matches expected SHA256")
                elif expected_sha256:
                    report_lines.append("  ⚠️ Does NOT match expected SHA256")

            # try to load checkpoint variables using TensorFlow
            if try_load:
                try:
                    import tensorflow as tf

                    ckpt_prefix = str(Path(root) / prefix)
                    report_lines.append(f"  Attempting to load checkpoint: {ckpt_prefix}")
                    ck = tf.train.load_checkpoint(ckpt_prefix)
                    vars_map = ck.get_variable_to_shape_map()
                    report_lines.append(f"  ✅ Loaded. Variables: {len(vars_map)}")
                    if show_all:
                        for name, shape in vars_map.items():
                            report_lines.append(f"    - {name}: {shape}")
                    else:
                        for i, (name, shape) in enumerate(vars_map.items()):
                            if i >= 20:
                                break
                            report_lines.append(f"    - {name}: {shape}")
                except Exception as e:
                    report_lines.append(f"  ❌ Failed to load checkpoint with TensorFlow: {e}")
                    report_lines.append("    (If you expect TF checkpoint loading to work, ensure TensorFlow is installed in this environment.)")

    return "\n".join(report_lines)

def _parse_args():
    p = argparse.ArgumentParser(
        description="Check TensorFlow checkpoint files under a weights directory."
    )
    p.add_argument("--dir", "-d", default="ImgCap/weights", help="Weights directory to scan")
    p.add_argument(
        "--expected-sha",
        "-s",
        default=None,
        help="Optional expected SHA256 (hex) to compare against each .data file",
    )
    p.add_argument(
        "--load",
        action="store_true",
        help="Try loading each checkpoint with TensorFlow (requires TF installed)",
    )
    p.add_argument(
        "--show-all",
        action="store_true",
        help="Show all variable names/shapes when loading a checkpoint (only with --load)",
    )
    p.add_argument(
        "--unpickle-tokenizer",
        action="store_true",
        help="If set, attempt to unpickle the tokenizer file to extract vocab info (unsafe: executes pickle).",
    )
    p.add_argument(
        "--out",
        "-o",
        default="tools/checkpoint_report.txt",
        help="Path to write the checkpoint report (default: tools/checkpoint_report.txt)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = check_tensorflow_weights(
        args.dir,
        expected_sha256=args.expected_sha,
        try_load=args.load,
        show_all=args.show_all,
        unpickle_tokenizer=args.unpickle_tokenizer,
    )
    out_path = Path(args.out)
    try:
        if report is None:
            print("No report generated.")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Report written to: {out_path}")
    except Exception as e:
        print(f"Failed to write report to {out_path}: {e}")
