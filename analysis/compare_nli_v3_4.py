"""Compare three development arms, common source/MR budgets and stronger text probes."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
from nli_v3_4.data import DATASETS, canonical_source, is_source, load_dataset, sha256, write_json
from analysis.hypothesis_only_logreg import fit_probe


def follows_by_key(rows):
    result={};counts=Counter()
    for row in rows:
        if not is_source(row):
            key=(row["pair_id"],row["mr_id"])
            result[(*key,counts[key])]=row;counts[key]+=1
    return result


def common_cohorts(left,right):
    if [canonical_source(r) for r in left if is_source(r)] != [canonical_source(r) for r in right if is_source(r)]:
        raise ValueError("original source cohorts differ")
    a,b=follows_by_key(left),follows_by_key(right)
    keys=sorted(a.keys()&b.keys())
    if any(a[k]["label"]!=b[k]["label"] or a[k]["premise"]!=b[k]["premise"] for k in keys):
        raise ValueError("matched requests have different labels/premises")
    return ([r for r in left if is_source(r)]+[a[k] for k in keys],
            [r for r in right if is_source(r)]+[b[k] for k in keys])


CUE_PATTERNS=(r"\bonly\s+if\b",r",\s*and\b",r",\s*but\b",r"\bbut\b",
 r"\b(?:no|not|never|none|nobody|nothing|neither|without|cannot)\b|\b\w+n['’]t\b",
 r"^\s*(?:no\b|nobody\b|nothing\b|none\b)",r"\bif\b",r"\b(?:may|might|maybe|perhaps|possibly)\b")


def inputs(rows,name):
    if name=="surface":
        return np.asarray([[int(bool(re.search(p,r["hypothesis"],re.I))) for p in CUE_PATTERNS]+
            [len(re.findall(r"\w+",r["hypothesis"])),len(r["hypothesis"]),r["hypothesis"].count(",")]
            for r in rows])
    return [r["hypothesis"] for r in rows]


def auxiliary(rows,tests):
    sources=[r for r in rows if is_source(r)];result=[]
    for seed in (42,43,44):
        fitted,_=train_test_split([r["pair_id"] for r in sources],test_size=.2,
                                  stratify=[r["label"] for r in sources],random_state=seed)
        fitted=set(fitted);training=[r for r in rows if r["pair_id"] in fitted]
        for name in ("word_svc","word_char_svc","surface"):
            word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=30000,sublinear_tf=True)
            if name=="surface":
                probe=Pipeline([("scale",StandardScaler()),("clf",LogisticRegression(C=1,max_iter=3000,random_state=seed))])
            else:
                features=word if name=="word_svc" else FeatureUnion([("word",word),
                    ("char",TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,max_features=60000,sublinear_tf=True))])
                probe=Pipeline([("features",features),("clf",LinearSVC(C=1,dual=True,max_iter=10000,random_state=seed))])
            with threadpool_limits(limits=2):
                probe.fit(inputs(training,name),[r["label"] for r in training])
            metrics={}
            for dataset,test in tests.items():
                follows=[r for r in test if not is_source(r)]
                metrics[dataset]=float(np.mean(probe.predict(inputs(follows,name))==[r["label"] for r in follows]))
            result.append({"seed":seed,"probe":name,"training_source_ids":sorted(fitted),"accuracy":metrics})
    return result


def run(root,output):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError(output)
    arms=("v3_3_original","v3_3","v3_4");loaded={};primary={};identities={}
    for arm in arms:
        loaded[arm]={d:load_dataset(root/arm/f"{d}.jsonl")[0] for d in ("train",*DATASETS)}
        primary[arm]=json.loads((root/f"{arm}_logreg/metrics.json").read_text())
        identities[arm]={d:sha256(root/arm/f"{d}.jsonl") for d in ("train",*DATASETS)}
    shared={d:common_cohorts(loaded["v3_3"][d],loaded["v3_4"][d]) for d in ("train",*DATASETS)}
    matched={}
    for index,arm in enumerate(("v3_3","v3_4")):
        probe,_=fit_probe(shared["train"][index]);matched[arm]={}
        for dataset in DATASETS:
            rows=[r for r in shared[dataset][index] if not is_source(r)]
            matched[arm][dataset]={"followups":len(rows),"accuracy":float(np.mean(
                probe.predict([r["hypothesis"] for r in rows])==[r["label"] for r in rows])) if rows else None}
    diagnostics=auxiliary(loaded["v3_4"]["train"],{d:loaded["v3_4"][d] for d in DATASETS})
    result={"stage":"development_only","input_hashes":identities,
            "primary":{arm:{d:m["evaluations"][d]["full"]["accuracy"] for d in DATASETS} for arm,m in primary.items()},
            "common_source_mr_ordinal":matched,"auxiliary_group_split_80_percent":diagnostics}
    write_json(output/"comparison.json",result)
    lines=["# v3.4 开发试验对照", "", "以下为开发来源上的重放试验，不能代替全量实际训练文件的正式确认。", "",
           "|来源|旧表达＋旧比例|旧表达＋均衡|新表达＋均衡|", "|---|---:|---:|---:|"]
    for d in DATASETS:lines.append("|"+d+"|"+"|".join(f"{result['primary'][a][d]:.2%}" for a in arms)+"|")
    lines.extend(["","共同来源／MR／同次序增强另见 comparison.json；两边使用相同原始行与共同增强预算。",
        "逐请求失败率、配额和MR覆盖见各生成报告。差值同时受样本分布和生成成功率影响。",
        "辅助探针按来源组80/20划分，种子42/43/44，只在训练部分拟合。"])
    (output/"REPORT_zh.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args();run(a.root,a.output)
