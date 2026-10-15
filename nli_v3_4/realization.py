"""Mechanical clause assembly, before judgment; never a factual rewrite."""
import re


CLAUSE_REALIZER_VERSION = "connector_initial_function_word_case_v1"


def realize_added_clause(hypothesis, connector, added_clause):
    """Join an edit plan, retaining names, acronyms, I and quoted text.

    Only unambiguous sentence-initial function words lose their initial capital.
    This does not repair fragments or infer omitted verbs/facts.
    """
    clause = added_clause.strip().rstrip(".!?")
    match = re.match(r"([A-Za-z]+)(?=\s)", clause)
    if match:
        word = match.group()
        pronouns = {"He", "She", "It", "They", "We", "You", "There"}
        # A capital following an article may belong to a name (The Beatles).
        article = word in {"A", "An", "The"} and re.match(r"\s+[a-z]", clause[match.end():])
        if word in pronouns or article:
            clause = word.lower() + clause[match.end():]
    return hypothesis.strip().rstrip(".!?") + connector + clause + "."
