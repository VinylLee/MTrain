"""Isolated round-4 development pilot. prepare/validate use zero model calls."""
import argparse
from collections import Counter,defaultdict
import fcntl
import json
from pathlib import Path

import run_nli_quality_repair_pilot as previous
from nli_v3_4 import core
from nli_v3_4.calls import _save
from nli_v3_4.data import digest,source_uid,sha256,read_jsonl,write_jsonl,export_source,export_followup
from nli_v3_4.protocol import has_api_failure
from nli_v3_4.redesign import Round4Engine,REVISION

ENGINE=Round4Engine()
KIND="round4_development_quality_pilot_draft1"


def code_hashes():
    return {**previous.code_hashes(),"nli_v3_4/redesign.py":sha256("nli_v3_4/redesign.py"),
            Path(__file__).name:sha256(__file__)}


def prepare(manifest_path,review_keys,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    _,sources=previous.frozen_sources(manifest_path)
    excluded={(r["dataset"],r["pair_id"]) for path in review_keys for r in read_jsonl(path)}
    requests=[]
    for origin in previous.ORIGINS:
        used=set()
        for operation in previous.OPERATIONS:
            candidates=[s for s in sources[origin] if (origin,s["pair_id"]) not in excluded
                        and s["pair_id"] not in used and ENGINE.eligible(s,operation)]
            candidates.sort(key=lambda s:digest([s["hypothesis"],operation,42,REVISION]))
            if len(candidates)<2:raise ValueError(f"insufficient unreviewed eligible sources: {origin}/{operation}")
            for source in candidates[:2]:
                used.add(source["pair_id"])
                requests.append({"request_id":digest([REVISION,origin,source_uid(source),operation,42]),
                    "dataset":origin,"pair_id":source["pair_id"],"source_uid":source_uid(source),
                    "operation":operation,"seed":42,"max_calls":9 if operation in core.COMPOSITE_STAGE_PLANS else 4})
    plan={"kind":KIND,"stage":"development","engine":"round4","revision":REVISION,
        "scope":"new_post_review_development_phase_not_frozen_or_production",
        "manifest":str(Path(manifest_path).resolve()),"manifest_sha256":sha256(manifest_path),
        "previous_review_keys":{str(Path(path).resolve()):sha256(path) for path in review_keys},
        "contract_sha256":ENGINE.fingerprint(),"contract":ENGINE.contract(),"code_hashes":code_hashes(),
        "requests":requests,"max_requests":50,"max_model_calls":250,"max_output_tokens":302000,
        "generation_tokens_per_call":1000,"judge_tokens_per_call":1400,"refill_rounds":0,
        "run_output":str((output.parent/"execution").resolve()),"production_allowed":False,
        "external_nli_in_generation":False}
    plan["plan_fingerprint"]=digest(plan);_save(output,plan);return plan


def validate(path):
    plan=json.loads(Path(path).read_text());claimed=plan.pop("plan_fingerprint")
    if digest(plan)!=claimed:raise ValueError("round-4 plan changed")
    plan["plan_fingerprint"]=claimed
    if (plan["kind"]!=KIND or plan["stage"]!="development" or plan["engine"]!="round4"
            or plan["contract_sha256"]!=ENGINE.fingerprint() or plan["contract"]!=ENGINE.contract()
            or plan["code_hashes"]!=code_hashes() or plan["production_allowed"] is not False
            or plan["external_nli_in_generation"] is not False or plan["refill_rounds"]!=0
            or plan["max_requests"]!=50 or plan["max_model_calls"]!=250
            or plan["max_output_tokens"]!=302000 or plan["generation_tokens_per_call"]!=1000
            or plan["judge_tokens_per_call"]!=1400):raise ValueError("round-4 contract/scope/budget changed")
    if sha256(plan["manifest"])!=plan["manifest_sha256"]:raise ValueError("frozen manifest changed")
    excluded=set()
    for key,h in plan["previous_review_keys"].items():
        if sha256(key)!=h:raise ValueError("previous review exposure changed")
        excluded.update((r["dataset"],r["pair_id"]) for r in read_jsonl(key))
    _,sources=previous.frozen_sources(plan["manifest"])
    by_id={d:{s["pair_id"]:s for s in rows} for d,rows in sources.items()};seen=set();strata=Counter()
    for r in plan["requests"]:
        key=(r["dataset"],r["pair_id"]);s=by_id[r["dataset"]][r["pair_id"]];op=r["operation"]
        if (key in seen or key in excluded or op not in previous.OPERATIONS or r["seed"]!=42
                or r["source_uid"]!=source_uid(s) or not ENGINE.eligible(s,op)
                or r["max_calls"]!=(9 if op in core.COMPOSITE_STAGE_PLANS else 4)
                or r["request_id"]!=digest([REVISION,r["dataset"],source_uid(s),op,42])):
            raise ValueError("invalid round-4 request")
        seen.add(key);strata[r["dataset"],op]+=1
    if (len(plan["requests"])!=50 or strata!=Counter({(d,op):2 for d in previous.ORIGINS for op in previous.OPERATIONS})
            or sum(r["max_calls"] for r in plan["requests"])!=250):raise ValueError("request budget or strata changed")
    return plan,sources


def run(path,*,resume=False,base_url="http://127.0.0.1:1234/v1"):
    plan,sources=validate(path);out=Path(plan["run_output"])
    if out.exists() and not resume:raise FileExistsError(out)
    if resume and not (out/"run.json").exists():raise FileNotFoundError("missing original run metadata")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config={"kind":KIND,"stage":"development","engine":"round4","plan_sha256":sha256(path),
            "contract_sha256":ENGINE.fingerprint(),"code_hashes":code_hashes(),"base_url":base_url,
            "manifest_sha256":plan["manifest_sha256"],"max_model_calls":250,"max_requests":50,
            "max_output_tokens":302000,"production_allowed":False}
        if resume:
            if json.loads((out/"run.json").read_text())!=config:raise ValueError("resume identity changed")
            entries=json.loads((out/"entries.json").read_text())
        else:_save(out/"run.json",config);entries=[];_save(out/"entries.json",entries)
        requests={r["request_id"]:r for r in plan["requests"]}
        if len({e["request_id"] for e in entries})!=len(entries):raise ValueError("duplicate saved request")
        by_id={d:{s["pair_id"]:s for s in rows} for d,rows in sources.items()}
        for e in entries:
            if e["request"]!=requests.get(e["request_id"]):raise ValueError("saved request identity changed")
            if e.get("row"):
                r=e["request"];s=by_id[r["dataset"]][r["pair_id"]]
                if ENGINE.replay_one(s,r["operation"],e["trace"],42)!=e["row"]["hypothesis"]:
                    raise ValueError("saved output not revalidated")
        completed={e["request_id"] for e in entries}
        def checkpoint(state):
            for d in previous.ORIGINS:
                schema="train" if d=="train" else "test";grouped=defaultdict(list)
                for e in entries:
                    if e.get("row") and e["request"]["dataset"]==d:grouped[e["request"]["pair_id"]].append(e["row"])
                rows=[]
                for s in sources[d]:rows.append(export_source(s,schema));rows.extend(grouped[s["pair_id"]])
                if d!="train":
                    for i,r in enumerate(rows):r["idx"]=i
                target=out/"v3_4"/f"{d}.jsonl";write_jsonl(target,rows)
                _save(target.with_suffix(".run.json"),{**config,"dataset":d,"source_sha256":digest(sources[d])})
            calls=list((out/"calls").glob("*/*.json"))
            if len(calls)>250:raise RuntimeError("call ceiling exceeded")
            _save(out/"status.json",{**config,"state":state,"requests_completed":len(entries),
                "outputs":sum(e.get("row") is not None for e in entries),"call_slots_consumed":len(calls),
                "requests_complete":len(entries)==50,"human_review_complete":False,"quality_passed":False,
                "goal_complete":False})
        checkpoint("prepared")
        if len(completed)==50:checkpoint("requests_complete");return
        from openai import OpenAI
        client=OpenAI(api_key="local",base_url=base_url,timeout=120,max_retries=0)
        state="interrupted";errors=0
        try:
            for r in plan["requests"]:
                if r["request_id"] in completed:continue
                s=by_id[r["dataset"]][r["pair_id"]]
                bounded=previous.BoundedRequestClient(client,out/"calls"/r["request_id"],r["max_calls"])
                text,trace=ENGINE.generate_one(bounded,s,r["operation"],42)
                label=core.LABEL_IDS[core.target_label(s,r["operation"])]
                row=export_followup(s,r["operation"],text,label,core.OPERATION_SPECS[r["operation"]]["mr_type"],
                    trace.get("applied_steps"),"train" if r["dataset"]=="train" else "test") if text else None
                entries.append({"request_id":r["request_id"],"request":r,"trace":trace,"row":row})
                _save(out/"entries.json",entries);completed.add(r["request_id"]);checkpoint("running")
                print(f"round4 requests={len(entries)}/50 outputs={sum(e.get('row') is not None for e in entries)}",flush=True)
                errors=errors+1 if has_api_failure(trace) else 0
                if errors>=3:state="service_error";raise RuntimeError("three API failures; preserve and stop")
            state="requests_complete"
        finally:checkpoint(state)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare");p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--previous-review-key",type=Path,action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("validate");p.add_argument("--plan",type=Path,required=True)
    p=sub.add_parser("run");p.add_argument("--plan",type=Path,required=True);p.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.command=="prepare":
        plan=prepare(args.manifest,args.previous_review_key,args.output)
        print(json.dumps({"prepared":True,"requests":50,"call_ceiling":250,"output_token_ceiling":302000,"api_calls":0}))
    elif args.command=="validate":validate(args.plan);print('{"valid":true,"api_calls":0}')
    else:run(args.plan,resume=args.resume)


if __name__=="__main__":main()
