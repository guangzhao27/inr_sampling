"""Extract metrics from offline W&B runs by scanning .wandb files for run names.

On this cluster W&B is always offline, and concurrent sbatch jobs corrupt
`coral/wandb/wandb/offline_run_paths.txt`, so the reliable way to find a run is to scan
every `offline-run-*/*.wandb` for the run_name byte-string. That is what this does.

Usage:
    # index every run whose name matches a prefix, dump the full history
    python script/inr_sample/extract_wandb_runs.py --prefix strat2d_ --out results.json

    # or resolve an explicit list of names
    python script/inr_sample/extract_wandb_runs.py --names a b c --out results.json

Writes JSON: {run_name: {"path": ..., "history": {metric: [[step, value], ...]},
                         "final": {metric: value}}}
and prints a summary table of the final values.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WANDB_DIR = REPO_ROOT / "coral" / "wandb" / "wandb"


def iter_records(wandb_file):
    """Yield parsed protobuf records from one .wandb file."""
    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_file))
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        yield rec


def read_run(wandb_file):
    """Return (run_name, history) for a .wandb file; history maps metric -> [(step, value)]."""
    run_name = None
    history = {}
    for rec in iter_records(wandb_file):
        kind = rec.WhichOneof("record_type")
        if kind == "run" and not run_name:
            run_name = rec.run.display_name or rec.run.run_id
        elif kind == "history":
            step = rec.history.step.num
            for item in rec.history.item:
                # Metrics are stored under `nested_key` (a repeated path), not `key`.
                name = item.key or ".".join(item.nested_key)
                if not name:
                    continue
                try:
                    value = json.loads(item.value_json)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    history.setdefault(name, []).append((step, value))
    return run_name, history


def scan(wandb_dir, name_filter):
    """Scan every offline run, keeping those whose display name passes `name_filter`."""
    out = {}
    files = sorted(Path(wandb_dir).glob("offline-run-*/*.wandb"))
    files += sorted(Path(wandb_dir).glob("run-*/*.wandb"))
    print(f"scanning {len(files)} .wandb files under {wandb_dir} ...", file=sys.stderr)

    for i, f in enumerate(files):
        if i % 200 == 0 and i:
            print(f"  ... {i}/{len(files)}", file=sys.stderr)
        # Cheap pre-filter: the run name is stored verbatim in the file.
        try:
            blob = f.read_bytes()
        except OSError:
            continue
        if not name_filter.get("prefix_bytes") or name_filter["prefix_bytes"] in blob:
            pass
        elif name_filter.get("names_bytes") and any(n in blob for n in name_filter["names_bytes"]):
            pass
        else:
            continue

        try:
            run_name, history = read_run(f)
        except Exception as exc:
            print(f"  ! failed to read {f}: {exc}", file=sys.stderr)
            continue
        if not run_name or not history:
            continue
        if not name_filter["match"](run_name):
            continue

        # A run name can appear in more than one file (re-runs); keep the newest.
        prev = out.get(run_name)
        if prev and prev["_mtime"] >= f.stat().st_mtime:
            continue
        final = {k: sorted(v)[-1][1] for k, v in history.items() if v}
        out[run_name] = {
            "path": str(f.relative_to(REPO_ROOT)),
            "_mtime": f.stat().st_mtime,
            "history": {k: sorted(v) for k, v in history.items()},
            "final": final,
        }
    for v in out.values():
        v.pop("_mtime", None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-dir", default=str(DEFAULT_WANDB_DIR))
    ap.add_argument("--prefix", help="keep runs whose name starts with this")
    ap.add_argument("--names", nargs="*", help="keep exactly these run names")
    ap.add_argument("--regex", help="keep runs whose name matches this regex")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="final_rel_loss",
                    help="metric shown in the printed summary")
    args = ap.parse_args()

    if not (args.prefix or args.names or args.regex):
        ap.error("one of --prefix / --names / --regex is required")

    rx = re.compile(args.regex) if args.regex else None
    names = set(args.names or [])

    def match(name):
        if names and name in names:
            return True
        if args.prefix and name.startswith(args.prefix):
            return True
        if rx and rx.search(name):
            return True
        return False

    name_filter = {
        "match": match,
        "prefix_bytes": args.prefix.encode() if args.prefix else None,
        "names_bytes": [n.encode() for n in names] if names else None,
    }
    if rx and not args.prefix and not names:
        name_filter["prefix_bytes"] = None  # no cheap pre-filter available; read all

    runs = scan(args.wandb_dir, name_filter)

    Path(args.out).write_text(json.dumps(runs, indent=1))
    print(f"\nresolved {len(runs)} runs -> {args.out}")
    if names:
        missing = sorted(names - set(runs))
        if missing:
            print(f"MISSING {len(missing)}: {missing}")

    metric_keys = sorted({k for r in runs.values() for k in r["final"]})
    key = args.key if args.key in metric_keys else (metric_keys[0] if metric_keys else None)
    if key:
        print(f"\n{'run_name':<62s} {key}")
        for name in sorted(runs):
            val = runs[name]["final"].get(key)
            print(f"{name:<62s} {val if val is None else f'{val:.6f}'}")
    print(f"\navailable metrics: {metric_keys}")


if __name__ == "__main__":
    main()
