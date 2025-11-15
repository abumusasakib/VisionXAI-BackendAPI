import os
import sys
import json
import base64
import subprocess
from pathlib import Path


# Ensure project root is on sys.path so `ImgCap` package can be imported
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

UPLOADED = ROOT / "uploaded_images"
OUT_DIR = ROOT / "tools"
OUT_DIR.mkdir(exist_ok=True)


def run_local():
    """Run inference in the local Python environment (importing ImgCap.captioner)."""
    # Import the captioner
    try:
        from ImgCap.captioner import generate_from_bytes
    except Exception as e:
        print(json.dumps({"error": f"Failed to import captioner: {e}"}))
        raise

    # Find a sample image
    imgs = [p for p in os.listdir(UPLOADED) if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not imgs:
        print(json.dumps({"error": "No images found in uploaded_images/"}))
        raise SystemExit(1)

    sample = UPLOADED / imgs[0]
    print(json.dumps({"using_image": str(sample)}, indent=2))

    with open(sample, "rb") as f:
        b = f.read()

    # Run generation
    out = generate_from_bytes(b)

    # If generation failed, print it
    if not isinstance(out, dict):
        print(json.dumps({"error": "generate_from_bytes returned non-dict", "value": str(out)}))
        raise SystemExit(1)

    # Save response.json for downstream tools and save attention image if present
    resp_path = OUT_DIR / "response.json"
    try:
        with open(resp_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
    except Exception as e:
        print(json.dumps({"warning": f"Failed to write response.json: {e}"}))

    # Save attention image if present
    att_b64 = out.get("attention_image")
    att_path = None
    if att_b64:
        try:
            imgdata = base64.b64decode(att_b64)
            att_path = OUT_DIR / "attention_out.png"
            with open(att_path, "wb") as f:
                f.write(imgdata)
        except Exception as e:
            print(json.dumps({"warning": f"Failed to write attention image: {e}"}))

    # Build summary: tokens + scores + attention shapes
    tokens = out.get("tokens", [])
    scores = out.get("token_scores", [])
    att_maps = out.get("attention_topk", [])
    att_means = out.get("attention_means", [])

    summary = {
        "caption": out.get("caption"),
        "num_tokens": len(tokens),
        "tokens": tokens,
        "scores": scores,
        "attention_means": att_means,
        "attention_topk": att_maps,
        "attention_image_path": str(att_path) if att_path else None,
        "response_json_path": str(resp_path),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Also print per-token attention lengths (if available via attention_topk or attention maps)
    if out.get("attention_topk"):
        print("attention_topk (per token):")
        for i, a in enumerate(out.get("attention_topk")):
            print(f" token {i}: {a}")

    print("Done.")


def build_and_run_docker(image_tag="visionxai-runner:latest"):
    """Build Docker image from repo root and run the inference inside it.

    The repo workspace is mounted into the container so outputs are written to the host.
    """
    # Build image
    print(f"Building Docker image '{image_tag}' (this may take a while)...")
    build_cmd = ["docker", "build", "-t", image_tag, "."]
    # Stream docker build output so the user can inspect progress and the font verification step
    proc = subprocess.Popen(build_cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.stdout:
        for line in proc.stdout:
            print(line, end="")
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"Docker build failed with exit code {proc.returncode}")

    # Run container: mount the host repo into /app so outputs persist
    # Use --rm to remove container after run
    host_path = str(ROOT.resolve())
    # On Windows, Docker accepts Windows paths for -v; keep as-is
    run_cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{host_path}:/app",
        image_tag,
        "python",
        "tools/run_inference_sample.py",
        "--in-docker",
    ]

    print("Running inference inside container...")
    proc = subprocess.run(run_cmd, cwd=str(ROOT), text=True)
    if proc.returncode != 0:
        raise SystemExit(f"Docker run failed with exit code {proc.returncode}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-docker", action="store_true", help="Build the project Docker image and run inference inside container")
    parser.add_argument("--in-docker", action="store_true", help="Indicates the script is running inside the container")
    args = parser.parse_args()

    if args.use_docker and not args.in_docker:
        build_and_run_docker()
        return

    # If running inside docker or not using docker, run local behavior
    run_local()


if __name__ == "__main__":
    main()
