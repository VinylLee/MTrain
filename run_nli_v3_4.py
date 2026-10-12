"""Local-only development/production CLI; external NLI scores never enter generation."""
from __future__ import annotations
import argparse
from pathlib import Path
from nli_v3_4.workflow import prepare, run_generation, freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-label", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser("generate")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dataset", choices=("train", "snli", "mnlim", "mnlimm", "sick"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=("development", "production"), default="development")
    p.add_argument("--version", choices=("v3.3", "v3.4"), default="v3.4")
    p.add_argument("--target-followups", type=int)
    p.add_argument("--quota-policy", choices=("balanced", "original"), default="balanced")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--limit-requests", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--freeze", type=Path)
    p = commands.add_parser("freeze")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--quality", type=Path, required=True)
    p.add_argument("--shortcut", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--round", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.train, args.test_root, args.output, args.per_label, args.seed)
    elif args.command == "generate":
        run_generation(args.manifest, args.dataset, args.output, stage=args.stage, version=args.version,
            target=args.target_followups, quota_policy=args.quota_policy, seed=args.seed,
            base_url=args.base_url, timeout=args.timeout, limit_requests=args.limit_requests,
            resume=args.resume, freeze_path=args.freeze)
    else:
        freeze(args.manifest, args.quality, args.shortcut, args.output, args.round)


if __name__ == "__main__":
    main()
