"""The preference-labelling web page (FEATURE_012, Phase 5a box 5).

A consumer of `soc_triage.rlhf`, never the other way round. This package imports
`rlhf.store` and reads the JSON `rlhf.pairs` writes; nothing in `rlhf` imports
anything from here. That keeps FEATURE_011's best property intact — the data
layer stays testable with no web framework installed at all.

`queue` and `render` are pure functions over dicts and import no framework
either. Only `app` needs FastAPI.
"""
