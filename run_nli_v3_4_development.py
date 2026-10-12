"""Resume the LOCAL development experiment. Stops at human review; never runs production."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from nli_v3_4 import core
from nli_v3_4.data import DATASETS, sha256, write_json

ROOT = Path(__file__).resolve().parent


def run(manifest, output, workers=2, design_python=None, round_number=1):
    if not 1 <= round_number <= 3 or workers not in (1,2):
        raise ValueError("at most three development rounds and two API workers")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity={"stage":"development","round":round_number,"manifest_sha256":sha256(manifest),
              "contract_sha256":core.fingerprint(),"production_allowed":False,"remote_push":False}
    identity_path=output/"experiment.json"
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError("round identity changed; preserve it and use a new round directory")
    write_json(identity_path,identity)
    status={**identity,"state":"running","jobs":{},"human_review_complete":False}

    def generate(arm,dataset):
        path=output/arm/f"{dataset}.jsonl";path.parent.mkdir(parents=True,exist_ok=True)
        version="v3.4" if arm=="v3_4" else "v3.3"
        quota="original" if arm=="v3_3_original" else "balanced"
        cmd=[sys.executable,str(ROOT/"run_nli_v3_4.py"),"generate","--manifest",str(manifest),
             "--dataset",dataset,"--version",version,"--quota-policy",quota,"--output",str(path)]
        if path.with_suffix(".run.json").exists():cmd.append("--resume")
        with (output/arm/f"{dataset}.stdout.log").open("a") as log:
            code=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        report=json.loads(path.with_suffix(".report.json").read_text()) if path.with_suffix(".report.json").exists() else {}
        return arm,dataset,code,report

    try:
        # Separate design experiment arms; none of these paths is a production pipeline.
        for arm in ("v3_4","v3_3","v3_3_original"):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending=[pool.submit(generate,arm,d) for d in ("train",*DATASETS)]
                for future in as_completed(pending):
                    _,dataset,code,report=future.result()
                    status["jobs"][f"{arm}/{dataset}"]={"exit_code":code,
                        "generation_complete":report.get("generation_complete",False),
                        "followups":report.get("followups",0)}
                    write_json(output/"status.json",status)
            if any(not status["jobs"][f"{arm}/{d}"]["generation_complete"] for d in ("train",*DATASETS)):
                status["state"]="partial_generation";write_json(output/"status.json",status)
                return 2
            if arm=="v3_4" and design_python:
                for dataset in ("train",*DATASETS):
                    diagnostic=output/"design_diagnostics"/dataset
                    if not (diagnostic/"report.json").exists():
                        subprocess.run([str(design_python),str(ROOT/"analysis/design_nli_quality.py"),
                            "--manifest",str(manifest),"--dataset",dataset,"--generated",str(output/arm/f"{dataset}.jsonl"),
                            "--output",str(diagnostic)],cwd=ROOT,check=True)
        for arm in ("v3_4","v3_3","v3_3_original"):
            if not (output/f"{arm}_logreg"/"metrics.json").exists():
                subprocess.run([sys.executable,str(ROOT/"analysis/hypothesis_only_logreg.py"),
                    "--train",str(output/arm/"train.jsonl"),"--test-root",str(output/arm),
                    "--output",str(output/f"{arm}_logreg")],cwd=ROOT,check=True)
        if not (output/"human_review"/"manifest.json").exists():
            subprocess.run([sys.executable,str(ROOT/"analysis/nli_v3_4_review.py"),"package",
                "--old-root",str(output/"v3_3"),"--new-root",str(output/"v3_4"),
                "--output",str(output/"human_review")],cwd=ROOT,check=True)
        if not (output/"comparison"/"comparison.json").exists():
            subprocess.run([sys.executable,str(ROOT/"analysis/compare_nli_v3_4.py"),
                "--root",str(output),"--output",str(output/"comparison")],cwd=ROOT,check=True)
        status["state"]="awaiting_human_review"
        status["shortcut_passed"]=json.loads((output/"v3_4_logreg/metrics.json").read_text())["all_datasets_pass"]
        write_json(output/"status.json",status)
        return 0
    except Exception as exc:
        status.update(state="failed",error=f"{type(exc).__name__}: {exc}")
        write_json(output/"status.json",status)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,choices=(1,2),default=2)
    p.add_argument("--design-python",type=Path);p.add_argument("--round",type=int,default=1)
    a=p.parse_args();return run(a.manifest.resolve(),a.output.resolve(),a.workers,
                              a.design_python.resolve() if a.design_python else None,a.round)


if __name__=="__main__":sys.exit(main())
