"""Second post-review protocol; explicit opt-in, preserving v1 behavior and results."""
import json
import re

from . import core, refinement as base
from .protocol import generator_json, ResponseFormatError

REVISION = "round4b_direct_typed_values_explicit_sentence_kinds_v2"
KINDS = "statement|fragment|question|command"
TRANSITIVE = ("buy|buys|bought|purchase|purchases|purchased|carry|carries|carried|hold|holds|held|"
    "watch|watches|watched|wear|wears|wore|read|reads|eat|eats|ate|drink|drinks|drank|"
    "throw|throws|threw|catch|catches|caught|kick|kicks|kicked|ride|rides|rode|wash|washes|"
    "cut|cuts|hit|hits|paint|paints|painted|use|uses|used|play|plays|played|touch|touches|help|helps")
PROGRESSIVE = "buying|purchasing|carrying|holding|watching|wearing|reading|eating|drinking|throwing|catching|kicking|riding|washing|cutting|hitting|painting|using|playing|touching|helping"
EXAMPLES = {"location": {"preposition": "inside", "place": "a building"},
    "position_or_posture": {"posture": "sitting"}, "activity": {"action_ing": "carrying a bag"},
    "possession": {"object_np": "a telescope"}, "material": {"material_np": "wood"},
    "shape_or_size": {"attribute": "tall"}, "colour": {"colour": "green"},
    "clothing": {"garment_np": "a coat"}, "count": {"count": 2, "objects_np": "books"},
    "time_or_duration": {"age_years": 35}}


def transitive_subset(hypothesis):
    # Preeligibility only: the exact transformation still needs the editor,
    # lexical/role guards, bidirectional NLI and grammar/operation judgments.
    # A visual-verb whitelist incorrectly excludes ordinary MNLI verbs.
    if hypothesis.strip().endswith(('?','!')):return False
    states='hungry|happy|sad|angry|tired|outside|inside|outdoors|indoors|here|there|red|blue|green'
    if re.match(r"(?i)^.{1,70}\b(?:is|are|was|were|isn't|aren't|wasn't|weren't)\s+(?:not\s+)?(?:"+states+r")\s*[.]?$",hypothesis.strip()):return False
    if re.match(r"(?i)^.{1,70}\b(?:runs?|walks?|sleeps?|stands?|sits?)\s+(?:fast|slowly|well|badly)\s*[.]?$",hypothesis.strip()):return False
    return True


def generation_schema(source, operation, round_index=0, seed=42):
    if operation not in base.parent.ADDITIONS:
        return base.generation_schema(source,operation,round_index,seed)
    domain=base.preference(source,round_index,seed)['domain']
    return base.fmt('one_hypothetical_property',base.obj({'detail':base.obj(base.DETAILS[domain])}))


def generation_prompt(source, operation, round_index=0, seed=42):
    if operation not in base.parent.ADDITIONS:
        return base.generation_prompt(source,operation,round_index,seed)
    pref=base.preference(source,round_index,seed);domain=pref['domain']
    return ("Propose one best-effort concrete attribute value in the required domain. "
        "This is a hypothetical candidate, not a factual answer: it need not be true of P. "
        "P may disagree or leave it unknown. Keep it compatible with H. "
        "Return only the typed detail, not an edited sentence. "
        "The code supplies the subject, polarity and finite verb; no pronouns or clauses in noun phrases. "
        "Subsequent independent checks reject unsuitable proposals, so no acceptance or abstention decision is requested here. "
        "Do not copy H or repair it.\n"
        +"Example output shape for this domain (choose your own suitable values): "+json.dumps({'detail':EXAMPLES[domain]})
        +"\nCURRENT INPUT:\n"+json.dumps({'premise':source['premise'],'source_hypothesis':source['hypothesis'],
            'entity':base.subject(source['hypothesis']),'preference':pref},ensure_ascii=False))


def parse_proposals(raw, source, operation, round_index=0, seed=42):
    if operation not in base.parent.ADDITIONS:
        return base.proposals(raw,source,operation,round_index,seed)
    value=generator_json(raw)
    if not isinstance(value,dict) or set(value)!={'detail'}:
        raise ResponseFormatError('expected exactly one typed hypothetical property')
    # No generated quote proxy. The four independent NLI relations still apply.
    wrapped=json.dumps({'abstain':False,'evidence_fact':'','detail':value['detail']})
    return base.proposals(wrapped,source,'uninformative',round_index,seed)


def edit_request(source, operation, proposal, trajectory=None):
    checks=base.COMMON+(base.ADDITION if operation in base.parent.ADDITIONS else ())
    if trajectory is not None:checks+=('ordered_components_valid',)
    sentence=base.obj({'kind':base.enum(KINDS),'grammatical':{'type':'boolean'}})
    other=[k for k in checks if k not in {'grammatical','source_is_complete_declarative','candidate_is_complete_declarative'}]
    schema=base.fmt('explicit_sentence_kinds_and_edit',base.obj({'source':sentence,'candidate':sentence,
        **{k:{'type':'boolean'} for k in other},'reason':base.TEXT}))
    payload={'source_text':source['hypothesis'],'candidate_text':proposal['hypothesis'],
        'claimed_edit':operation,'definition':core.OPERATION_SPECS[operation]['instruction']}
    if operation in base.parent.ADDITIONS:payload['proposed_detail']=proposal
    if trajectory is not None:payload['ordered_trajectory']=trajectory
    prompt=("Classify each text's kind separately from its grammar. statement means a complete assertion "
        "with a subject and finite verb, even without final punctuation. fragment lacks a complete assertion. "
        "Questions and commands are separate. A statement can still be ungrammatical: "
        "'The men is outside' has kind=statement but grammatical=false. "
        "'Her team is winning.' has kind=statement and grammatical=true. "
        "'A man in a red shirt.' has kind=fragment. Do not confuse formal wording with incompleteness. "
        "Then check the actual edit, without NLI labels. No generator claim proves correctness.\n"
        +json.dumps(payload,ensure_ascii=False))
    return prompt,schema,checks


class DirectTypedEngine(base.RefinementEngine):
    revision=REVISION
    generation_system="Propose one hypothetical property value about the supplied entity. Keep H fixed. Return the requested typed JSON. Candidate acceptance is checked separately; do not predict an NLI label."
    generation_prompt=staticmethod(generation_prompt)
    generation_schema=staticmethod(generation_schema)
    parse_proposals=staticmethod(parse_proposals)
    edit_request=staticmethod(edit_request)

    @staticmethod
    def contract():
        return {**base.RefinementEngine.contract(),'revision':REVISION,
            'generation_system':DirectTypedEngine.generation_system,
            'parent_refinement_contract':base.RefinementEngine.fingerprint(),
            'proposal_protocol':'typed_best_effort_no_abstention_field_no_quote_proxy',
            'edit_protocol':'separate_sentence_kind_and_grammar_no_NLI_labels',
            'voice_preeligibility':'prune_obvious_copular_states_questions_and_bare_motion_not_a_transitive_proof',
            'domain_shape_examples':EXAMPLES,
            'independent_contradiction_rule':'P_H_E_P_candidate_C_P_added_C_H_added_N_unchanged'}

    def eligible(self,source,operation):
        if operation=='voice_switch':return core.eligible(source,operation) and transitive_subset(source['hypothesis'])
        if operation in base.parent.ADDITIONS:
            entity=base.subject(source['hypothesis'])
            if not entity or re.search(r"(?i)\b(because|if|while|when|would|could|should|may|might|must|will|can|did|does|what|who|where|why|how)\b|['’](?:d|ll|ve|re)\b",entity['anchor']):
                return False
        return super().eligible(source,operation)

    def normalize_edit(self,value,checks):
        if not isinstance(value,dict):raise ResponseFormatError('missing edit object')
        other=set(checks)-{'grammatical','source_is_complete_declarative','candidate_is_complete_declarative'}
        if set(value)!={'source','candidate','reason'}|other or not isinstance(value['reason'],str):
            raise ResponseFormatError('edit fields differ from explicit sentence schema')
        for key in ('source','candidate'):
            v=value[key]
            if (not isinstance(v,dict) or set(v)!={'kind','grammatical'} or v['kind'] not in KINDS.split('|')
                    or type(v['grammatical']) is not bool):raise ResponseFormatError('invalid sentence kind or grammar verdict')
        return {**{k:value[k] for k in other},'reason':value['reason'],
            'grammatical':value['source']['grammatical'] and value['candidate']['grammatical'],
            'source_is_complete_declarative':value['source']['kind']=='statement',
            'candidate_is_complete_declarative':value['candidate']['kind']=='statement'}
