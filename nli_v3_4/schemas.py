"""Mechanical output schemas; no NLI predictions or source labels are consulted."""
from .invariants import WORD

GENERATION_FORMAT = "json_schema_atomic_INV_v1"


class FormattedClient:
    """Constrain one generation call; judgment calls keep their original format."""
    def __init__(self, client, response_format):
        self.client = client
        self.response_format = response_format
        self.chat = self.completions = self

    def create(self, **kwargs):
        if "response_format" in kwargs:
            raise RuntimeError("generation response format supplied twice")
        return self.client.chat.completions.create(**kwargs, response_format=self.response_format)


def invariant_generation_schema(source,operation):
    marker={"type":"string","enum":["inherit_source"]}
    if operation=="synonym_replacement":
        slots=[]
        for index,match in enumerate(WORD.finditer(source["hypothesis"])):
            slots.append({"type":"object","properties":{
                "word_index":{"type":"integer","enum":[index]},
                "old_word":{"type":"string","enum":[match.group()]},
                "replacement":{"type":"string"}},
                "required":["word_index","old_word","replacement"],"additionalProperties":False})
        if not slots:
            return {"type":"json_schema","json_schema":{"name":"invariant_candidates","strict":True,
                    "schema":{"type":"array","maxItems":0}}}
        edits={"type":"array","minItems":1,"maxItems":2,"items":{"oneOf":slots}}
        plan={"type":"object","properties":{"edits":edits},"required":["edits"],"additionalProperties":False}
        item={"type":"object","properties":{"label_rule":marker,"edit_plan":plan},
              "required":["label_rule","edit_plan"],"additionalProperties":False}
    elif operation in {"voice_switch","pronoun_substitution"}:
        item={"type":"object","properties":{"label_rule":marker,"hypothesis":{"type":"string"}},
              "required":["label_rule","hypothesis"],"additionalProperties":False}
    else:
        raise ValueError("output schema is restricted to invariant operations")
    return {"type":"json_schema","json_schema":{"name":"invariant_candidates","strict":True,
            "schema":{"type":"array","minItems":0,"maxItems":3,"items":item}}}
