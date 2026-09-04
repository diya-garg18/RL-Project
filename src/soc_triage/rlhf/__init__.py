"""RLHF data layer (Phase 5a) — FEATURE_011.

Four modules, none of which import an agent, torch, or the environment:

    summary.py    EpisodeRecord -> a summary a person can judge
    pairs.py      records -> blinded preference pairs
    store.py      SQLite for the labels people give back
    agreement.py  Cohen's kappa over the double-labelled pairs

The deliberate absence of those imports is the design (FEATURE_011 §3). Pair
construction reads EpisodeRecord JSON off disk rather than running policies, so
this whole layer works on a clone with no `results/` and no `torch` — and so
nothing here can be dragged into an earlier phase's dependency graph
(CONSTRAINTS #11).
"""
