"""Crash recovery must preserve responses and consume uncertain call slots."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from nli_v3_4 import core
from nli_v3_4.calls import DurableRequestClient
from nli_v3_4.protocol import InterruptedCallError, exception_fields, has_api_failure


def completion(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def backend(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_killed_process_replays_response_and_does_not_resend_inflight_call(tmp_path):
    project = Path(__file__).resolve().parents[1]
    harness = tmp_path / "harness.py"
    harness.write_text('''import sys,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1])
from nli_v3_4.calls import DurableRequestClient
root=Path(sys.argv[2])
def create(**kwargs):
    prompt=kwargs['prompt']
    with (root/'physical_calls').open('a') as log:
        log.write(prompt+'\\n');log.flush()
    if prompt=='second':
        (root/'second_started').touch()
        time.sleep(120)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='saved first response'))])
api=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
client=DurableRequestClient(api,root/'calls')
client.create(prompt='first')
client.create(prompt='second')
''')
    with (tmp_path / "harness.log").open("w") as log:
        child = subprocess.Popen([sys.executable, str(harness), str(project), str(tmp_path)],
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 10
            while not (tmp_path / "second_started").exists() and time.monotonic() < deadline:
                assert child.poll() is None, (tmp_path / "harness.log").read_text()
                time.sleep(.05)
            assert (tmp_path / "second_started").exists()
            os.kill(child.pid, signal.SIGKILL)
            assert child.wait(timeout=5) == -signal.SIGKILL
        finally:
            if child.poll() is None:
                os.kill(child.pid, signal.SIGKILL)
                child.wait()
    assert (tmp_path / "physical_calls").read_text().splitlines() == ["first", "second"]
    called = []
    client = DurableRequestClient(backend(lambda **kw: called.append(kw) or completion("unused budget")),
                                  tmp_path / "calls")
    assert client.create(prompt="first").choices[0].message.content == "saved first response"
    with pytest.raises(InterruptedCallError):
        client.create(prompt="second")
    assert called == []
    assert client.create(prompt="third").choices[0].message.content == "unused budget"
    assert len(called) == 1
    assert (client.new_calls, client.replayed_calls, client.uncertain_calls) == (1, 1, 1)


def test_atomic_generator_uses_only_remaining_round_after_interrupted_judge(tmp_path):
    source = {"premise": "A man carries a box.", "hypothesis": "A man carries a box.",
              "label": 0, "pair_id": 7}
    candidate = {"label_rule": "inherit_source", "edit_plan": {"edits": [
        {"word_index": 2, "old_word": "carries", "replacement": "transports"}]}}
    verdict = {"index": 0, "source_to_candidate": "entailment",
               "candidate_to_source": "entailment", "grammatical": True,
               "operation_valid": True, "source_equivalent": True}
    physical = []
    def interrupted(**kwargs):
        physical.append(kwargs)
        if len(physical) == 1:
            return completion(json.dumps([candidate]))
        raise KeyboardInterrupt("judge interrupted")
    with pytest.raises(KeyboardInterrupt):
        core.generate_atomic(DurableRequestClient(backend(interrupted), tmp_path / "calls"),
                             source, "synonym_replacement")
    def continued(**kwargs):
        physical.append(kwargs)
        return completion(json.dumps([candidate] if len(physical) == 3 else [verdict]))
    client = DurableRequestClient(backend(continued), tmp_path / "calls")
    text, trace = core.generate_atomic(client, source, "synonym_replacement")
    assert text == "A man transports a box."
    assert len(physical) == 4  # Two original calls plus only the unused second round.
    assert len(trace["attempts"]) == 2
    assert trace["attempts"][0]["error_kind"] == "interrupted_call"
    assert (client.new_calls, client.replayed_calls, client.uncertain_calls) == (2, 1, 1)


def test_call_identity_change_fails_without_overwriting_or_issuing_api(tmp_path):
    calls = []
    api = backend(lambda **kw: calls.append(kw) or completion("raw original"))
    DurableRequestClient(api, tmp_path).create(prompt="original")
    before = (tmp_path / "0000.json").read_bytes()
    with pytest.raises(RuntimeError, match="identity changed"):
        DurableRequestClient(api, tmp_path).create(prompt="changed")
    assert len(calls) == 1 and (tmp_path / "0000.json").read_bytes() == before


def test_uncertain_call_has_its_own_error_category():
    fields = exception_fields(InterruptedCallError("no recorded response"))
    assert fields["error_kind"] == "interrupted_call"
    assert not has_api_failure(fields)


def test_unknown_interruption_reservation_cannot_produce_a_row_or_resume_calls():
    from copy import deepcopy
    from nli_v3_4.calls import validate_budget_reservation
    from nli_v3_4.data import digest, source_uid
    source = {"premise":"A man carries a box.","hypothesis":"A man carries a box.","label":1,"pair_id":7}
    entry = {"budget_reservation":True,"row":None,"source_uid":source_uid(source),
             "operation":"composite_inv","refill_round":1,
             "trace":{"status":"interrupted_uncertain_budget_reserved",
                      "actual_model_calls_known":False,"model_response_available":False}}
    entry['request_id'] = digest([source_uid(source),'composite_inv',1,42,'v3.4'])
    validate_budget_reservation(entry,source,42,'v3.4')
    for patch in ({"row":{}},{"pending_continuation":True},{"budget_reservation":False},
                  {"request_id":"wrong"},{"source_uid":"wrong"}):
        changed = {**deepcopy(entry),**patch}
        with pytest.raises(RuntimeError):validate_budget_reservation(changed,source,42,'v3.4')


def test_only_generation_has_schema_judgment_labels_remain_unconstrained():
    from nli_v3_4.schemas import invariant_generation_schema
    source = {"premise":"A man carries a box.","hypothesis":"A man carries a box.","label":1,"pair_id":7}
    value = {"label_rule":"inherit_source","edit_plan":{"edits":[
        {"word_index":2,"old_word":"carries","replacement":"transports"}]}}
    verdict = {"index":0,"source_to_candidate":"entailment","candidate_to_source":"entailment",
               "source_equivalent":True,"grammatical":True,"operation_valid":True}
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls)==1:
            assert kwargs['response_format']==invariant_generation_schema(source,'synonym_replacement')
            assert kwargs['temperature']==.7 and kwargs['max_tokens']==700
            return completion(json.dumps([value]))
        assert 'response_format' not in kwargs
        assert kwargs['temperature']==0 and kwargs['max_tokens']==600
        return completion(json.dumps([verdict]))
    text,_ = core.generate_atomic(backend(create),source,'synonym_replacement')
    assert text=='A man transports a box.' and len(calls)==2


def test_legacy_call_metadata_migration_remains_resumable(tmp_path, monkeypatch):
    import openai
    from nli_v3_4 import workflow
    from nli_v3_4.calls import CALL_PROTOCOL
    from nli_v3_4.data import DATASETS, write_json, write_jsonl
    from nli_v3_4.recovery import migrate
    sources = [{"premise": "A man carries a box.", "hypothesis": "A man carries a box.",
                "label": label, "pair_id": label} for label in range(3)]
    original = tmp_path / "original.jsonl"
    tests = tmp_path / "original_tests"
    write_jsonl(original, sources)
    for dataset in DATASETS:
        write_jsonl(tests / f"{dataset}.jsonl", sources)
    workflow.prepare(original, tests, tmp_path / "frozen", per_label=1)
    manifest = tmp_path / "frozen/manifest.json"
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(core.legacy, "_completion", lambda *args, **kwargs: "OK")
    old = tmp_path / "old"
    for dataset in ("train", *DATASETS):
        output = old / "v3_4" / f"{dataset}.jsonl"
        workflow.run_generation(manifest, dataset, output, target=3, limit_requests=0)
        metadata = output.with_suffix(".run.json")
        record = json.loads(metadata.read_text())
        record.pop("call_protocol")
        write_json(metadata, record)
    recovered = tmp_path / "recovered"
    migrate(old, manifest, recovered)
    output = recovered / "v3_4/train.jsonl"
    assert json.loads(output.with_suffix(".run.json").read_text())["call_protocol"] == CALL_PROTOCOL
    workflow.run_generation(manifest, "train", output, target=3, limit_requests=0, resume=True)

    # A migration must not abandon a dispatched request that has not reached
    # the logical journal. Its original snapshot must finish recovery first.
    orphan = (old / "v3_4/train.jsonl").with_suffix(".calls") / "unrecorded_request"
    orphan.mkdir(parents=True)
    with pytest.raises(ValueError, match="original code"):
        migrate(old, manifest, tmp_path / "unsafe_migration")
