"""Bounded xdog-ai invocations; no reference uploads or authentication handling."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path("/home/zhang/workspace/zdog-yizhu")
DEST = ROOT / "client/design/generated/tabs-login-v1"
CLI = "/home/zhang/workspace/xdog/.venv/bin/xdog-ai"
SPECS = {
    "home-concept": ("9:16", "1K"),
    "tabs-overview": ("3:2", "2K"),
    "login-flow": ("4:3", "2K"),
}

parser = argparse.ArgumentParser()
parser.add_argument("asset", choices=SPECS)
args = parser.parse_args()
aspect, size = SPECS[args.asset]
prompt_path = DEST / f"{args.asset}.prompt.txt"
record_path = DEST / f"{args.asset}.generation.json"
if record_path.exists():
    sys.exit("Existing generation record; refusing an implicit retry.")
output_dir = DEST / "raw" / args.asset
output_dir.mkdir(parents=True, exist_ok=False)
command = [
    CLI, "image", "antigravity", "gemini-3.1-flash-image",
    prompt_path.read_text(), "--output-dir", str(output_dir),
    "--aspect-ratio", aspect, "--image-size", size, "--json",
]
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
record = {
    "asset": args.asset,
    "provider": "antigravity",
    "model": "gemini-3.1-flash-image",
    "method": "xdog-ai image; text-to-image; no uploaded references",
    "prompt": str(prompt_path.relative_to(ROOT)),
    "aspect_ratio_requested": aspect,
    "image_size_requested": size,
    "started_at": started,
    "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "returncode": result.returncode,
    "status": "draft / not approved for integration",
    "photography": "AI concept imagery, not real observations",
}
try:
    response = json.loads(result.stdout)
except json.JSONDecodeError:
    response = {}
paths = response.get("paths", []) if isinstance(response, dict) else []
record["paths"] = paths
if result.returncode or not paths:
    record["status"] = "failed / no confirmed generated image"
# Persist only minimal provenance, never entire provider payloads or headers.
with record_path.open("x") as stream:
    json.dump(record, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps(record, ensure_ascii=False, indent=2))
if result.returncode or not paths:
    print("No confirmed successful image paths. Inspect saved files before retrying.")
    # Report known failure classes without printing potentially sensitive logs.
    error = (result.stderr + result.stdout).lower()
    for marker in ("permission denied", "timed out", "timeout", "unauthorized",
                   "expired", "login", "connection", "resolve", "quota",
                   "rate limit", "unsupported", "resource_exhausted"):
        if marker in error:
            print("Failure marker:", marker)
    sys.exit(1)
