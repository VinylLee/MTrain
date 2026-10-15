"""Isolated engineering-repair profile using the existing bounded pilot runner.

Importing this module does not change the original runner or old experiment code
hashes. Configuration is scoped to a single command and restored on return.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import run_nli_v3_4_refinement as pilot
from nli_v3_4.article_guard import ArticleCheckedEngine
from nli_v3_4.data import sha256


@contextmanager
def configured_pilot():
    keys = ("ENGINE", "KIND", "REVISION", "ENGINE_NAME", "code_hashes")
    original = {key: getattr(pilot, key) for key in keys}
    original_hashes = original["code_hashes"]
    def hashes():
        return {**original_hashes(), "nli_v3_4/article_guard.py": sha256("nli_v3_4/article_guard.py"),
                Path(__file__).name: sha256(__file__)}
    try:
        pilot.ENGINE = ArticleCheckedEngine()
        pilot.KIND = "post_review_v2_article_guard_engineering_repair_pilot_v1"
        pilot.REVISION = pilot.ENGINE.revision
        pilot.ENGINE_NAME = "refinement_v2_article"
        pilot.code_hashes = hashes
        yield pilot
    finally:
        for key, value in original.items():
            setattr(pilot, key, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--previous-review-key", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    for command in ("validate", "run", "audit"):
        p = sub.add_parser(command)
        p.add_argument("--plan", type=Path, required=True)
        if command == "run":
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with configured_pilot() as runner:
        if args.command == "prepare":
            runner.prepare(args.manifest, args.previous_review_key, args.output)
            print('{"prepared":true,"api_calls":0}')
        elif args.command == "validate":
            runner.validate(args.plan)
            print('{"valid":true,"api_calls":0}')
        elif args.command == "audit":
            print(json.dumps(runner.audit(args.plan), ensure_ascii=False))
        else:
            runner.run(args.plan, args.resume)


if __name__ == "__main__":
    main()
