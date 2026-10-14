"""Development-only backend syntax probe; never exports accepted augmentation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from openai import OpenAI
from nli_v3_4 import core
from nli_v3_4.data import write_json,sha256
from nli_v3_4.schemas import invariant_generation_schema
from nli_v3_4.workflow import source_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--dataset',choices=('train','snli','mnlim','mnlimm','sick'),required=True)
    p.add_argument('--pair-id',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    active=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:command=(proc/'cmdline').read_bytes().decode().replace('\0',' ')
        except (FileNotFoundError,PermissionError):continue
        if 'run_nli_v3_4.py generate ' in command:active.append(int(proc.name))
    if len(active)>=2:raise RuntimeError(f'two generation workers active; defer schema probe: {active}')
    _,_,sources=source_rows(a.manifest,a.dataset,'development')
    source=next(r for r in sources if r['pair_id']==a.pair_id)
    schema=invariant_generation_schema(source,'synonym_replacement')
    client=OpenAI(api_key='local',base_url='http://127.0.0.1:1234/v1',timeout=120,max_retries=0)
    result={'stage':'development_backend_syntax_only','accepted_rows':0,'production_allowed':False,
            'manifest_sha256':sha256(a.manifest),'dataset':a.dataset,'pair_id':a.pair_id,'response_format':schema}
    try:
        response=client.chat.completions.create(model=core.ALLOWED_MODEL,messages=[
            {'role':'system','content':core.INVARIANT_SYSTEM},
            {'role':'user','content':core.build_generation_prompt(source,'synonym_replacement')}],
            temperature=.7,max_tokens=700,response_format=schema)
        raw=response.choices[0].message.content or '';result['raw_response']=raw
        value=json.loads(raw)
        import jsonschema
        jsonschema.validate(value,schema['json_schema']['schema'])
        result.update(schema_valid=True,parser_candidates=core.parse_candidates(raw,source,'synonym_replacement'),
                      note='No semantic judgment performed; parser candidates are not accepted data.')
    except Exception as exc:
        result.update(schema_valid=False,error=f'{type(exc).__name__}: {exc}')
    write_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('response_format','raw_response')}))


if __name__=='__main__':main()
