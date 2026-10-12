"""Read-only local progress, including live checkpoints before a dataset finishes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def status(root):
    root=Path(root)
    driver=json.loads((root/"driver.json").read_text()) if (root/"driver.json").exists() else {}
    pid=driver.get("pid")
    try:
        command=Path(f"/proc/{pid}/cmdline").read_bytes().decode().replace("\x00"," ")
        alive="run_nli_v3_4_development.py" in command
    except (FileNotFoundError,PermissionError):alive=False
    summary=json.loads((root/"status.json").read_text()) if (root/"status.json").exists() else {}
    jobs={}
    for arm in ("v3_4","v3_3","v3_3_original"):
        for dataset in ("train","snli","mnlim","mnlimm","sick"):
            path=root/arm/f"{dataset}.report.json"
            if not path.exists():
                jobs[f"{arm}/{dataset}"]={"state":"pending","followups":0,"target":180}
            else:
                m=json.loads(path.read_text())
                jobs[f"{arm}/{dataset}"]={"state":m["state"],"followups":m["followups"],
                    "target":m["target_followups"],"requests":m["requests"],"label_counts":m["label_counts"],
                    "generation_complete":m["generation_complete"]}
    return {"driver_pid":pid,"driver_alive":alive,
        "state":summary.get("state","running" if alive else "not_running"),
        "jobs":jobs,"overall_complete":False,"local_only":True,
        "human_review_complete":summary.get("human_review_complete",False),
        "shortcut_passed":summary.get("shortcut_passed"),
        "error":summary.get("error")}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True)
    a=p.parse_args();print(json.dumps(status(a.root),ensure_ascii=False,indent=2))
