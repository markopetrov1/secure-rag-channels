"""Canonical arm identifiers and display names.

Four scripts previously carried their own copy of this map, which is how a
variant arm came to be missing from all of them: a runner writes a variant as
its own arm token so the variant survives the underscore-splitting that recovers
track, mode, arm and generator from a filename, but no consumer knew the name,
so the rows would have arrived with no label and no setup cost attributed.

An arm's *variant* carries its own results while sharing the *base* arm's
one-time setup expenditure. a7s1 and a7s2 are extra optimiser seeds that share
the compile budget accounted to a7, and a8c shares the corpus embedding a8
already pays for.
"""
import re

NAMES = {
    "a1": "Zero-shot closed book",
    "a2o": "One-shot ICL closed book",
    "a2": "Few-shot ICL closed book",
    "a3": "Long context",
    "a4": "Dense RAG",
    "a5": "Hybrid RAG (RRF)",
    "a7": "DSPy-optimized RAG",
    "a8": "Agentic ReAct RAG",
    "a8c": "Agentic critique-revise RAG",
    "a7s1": "DSPy-optimized RAG, second optimiser seed",
    "a7s2": "DSPy-optimized RAG, third optimiser seed",
}

# Short forms for figure axes and narrow table columns.
SHORT = {
    "a1": "Zero-shot", "a2o": "One-shot ICL",
    "a2": "Few-shot ICL", "a3": "Long context",
    "a4": "Dense RAG", "a5": "Hybrid RAG", "a7": "DSPy RAG",
    "a8": "Agentic ReAct", "a8c": "Agentic critique",
    "a7s1": "DSPy RAG (seed 2)", "a7s2": "DSPy RAG (seed 3)",
}

# What each base arm pays for once, before any question is asked.
SETUP_KINDS = {
    "a4": ("corpus embedding",),
    "a5": ("corpus embedding",),
    "a7": ("corpus embedding", "prompt compilation"),
    "a8": ("corpus embedding",),
}


# Arms that share a hue with a base arm in figures, because they are variants of
# it rather than separate paradigms. The one-shot arm is a variant of few-shot.
HUE_OF = {"a2o": "a2", "a8c": "a8"}


def base_arm(arm):
    """The arm whose setup cost this variant draws on.

    a7s1 and a7s2 are further seeds of a7's compile and a8c reuses the corpus
    embedding a8 pays for, so both charge against the base arm rather than
    reporting a setup cost of zero.
    """
    arm = re.sub(r"s\d+$", "", str(arm))
    if arm.startswith("a8"):
        return "a8"
    return arm


def name(arm, short=False):
    table = SHORT if short else NAMES
    arm = str(arm)
    return table.get(arm) or table.get(base_arm(arm)) or arm
