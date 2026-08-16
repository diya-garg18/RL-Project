"""Render the learned policy as a readable table (ROADMAP Phase 2, box 6).

A 576-row Q-table is not a result anyone can read. This turns it into the
question the project actually cares about: **as the shift runs out, does the
agent change its strategy — and in a way a human analyst would recognise?**

Two views, both written to results/policy_table.md:

  1. **Full grid** — three panels, one per `time_left` bucket, each 16 rows
     (queue_len x oldest_age) by 12 columns (max_severity x asset_criticality).
     16 x 12 x 3 = 576, so nothing is aggregated away and nothing is hidden.
  2. **Action shares per time bucket**, weighted by visit count — the summary
     that answers the strategy-shift question directly.

**The unvisited-state problem, and why this script is careful about it.**
An unvisited state has an all-zero Q row. `argmax` then returns action 0 by the
tie-break rule, which is *not* a decision the agent ever made. Printed naively,
the hundreds of states Q-learning never reaches would render as a confident
preference for PULL_HIGHEST_SEVERITY — a figure that misleads a reader and an
examiner equally. Every unvisited cell is therefore printed as `·`, and the
share table counts only states the agent actually visited.

Usage:
    python scripts/policy_table.py          # after scripts/train.py has run
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_triage.config import load_env_config

# Single characters so a 12-column grid stays readable at a glance.
ACTION_CODES = {
    0: "S",  # PULL_HIGHEST_SEVERITY
    1: "O",  # PULL_OLDEST
    2: "A",  # PULL_MOST_CRITICAL_ASSET
    3: "C",  # PULL_CHEAPEST
    4: "B",  # BULK_CLOSE_LOW_RISK
}
UNVISITED = "·"

N_SEV, N_QLEN, N_AGE, N_TLEFT, N_CRIT = 4, 4, 4, 3, 3


def decode(state_id: int) -> tuple[int, int, int, int, int]:
    """Invert the mixed-radix packing in `state.discretise`.

    discretise builds:  ((((sev*4 + qlen)*4 + age)*3 + tleft)*3 + crit
    so unpacking runs the digits off the bottom in reverse order. Written as an
    explicit unpack rather than divmod chaining, because getting this backwards
    silently transposes the entire figure.
    """
    crit = state_id % N_CRIT
    state_id //= N_CRIT
    tleft = state_id % N_TLEFT
    state_id //= N_TLEFT
    age = state_id % N_AGE
    state_id //= N_AGE
    qlen = state_id % N_QLEN
    state_id //= N_QLEN
    sev = state_id
    return sev, qlen, age, tleft, crit


def encode(sev: int, qlen: int, age: int, tleft: int, crit: int) -> int:
    """Forward packing — kept beside `decode` so the pair can be checked against
    each other, which `_self_check` does before anything is printed."""
    return (((sev * N_QLEN + qlen) * N_AGE + age) * N_TLEFT + tleft) * N_CRIT + crit


def _self_check() -> None:
    """encode/decode must be exact inverses over all 576 ids.

    Cheap, and it runs every time: a transposed figure is the kind of error that
    survives review because the table still *looks* plausible.
    """
    for state_id in range(N_SEV * N_QLEN * N_AGE * N_TLEFT * N_CRIT):
        if encode(*decode(state_id)) != state_id:
            raise SystemExit(f"decode/encode disagree at state {state_id} — STOP")


def bucket_labels(boundaries: tuple[float, ...], unit: str = "") -> list[str]:
    """Human-readable range labels from the ascending boundaries in config."""
    labels = [f"<{boundaries[0]:g}{unit}"]
    for low, high in zip(boundaries, boundaries[1:]):
        labels.append(f"{low:g}-{high:g}{unit}")
    labels.append(f">{boundaries[-1]:g}{unit}")
    return labels


def greedy_with_unvisited(Q: np.ndarray, visits: np.ndarray) -> list[str]:
    """Action code per state, or `·` where the agent never acted.

    Ties break to the lower index, matching QLearningAgent._argmax and
    agents.dp.greedy_policy — so this table and the DP policy table are
    comparable rather than merely similar.
    """
    codes = []
    for state in range(Q.shape[0]):
        if visits[state].sum() == 0:
            codes.append(UNVISITED)
            continue
        best_a, best_v = 0, -np.inf
        for action in range(Q.shape[1]):
            if Q[state, action] > best_v:
                best_v, best_a = Q[state, action], action
        codes.append(ACTION_CODES[best_a])
    return codes


def main() -> None:
    _self_check()

    q_path = ROOT / "results" / "q_learning_Q.npy"
    v_path = ROOT / "results" / "q_learning_visits.npy"
    if not q_path.exists() or not v_path.exists():
        raise SystemExit(
            "results/q_learning_Q.npy or q_learning_visits.npy missing — "
            "run `python scripts/train.py` first (results/ is gitignored)."
        )

    cfg = load_env_config(ROOT / "config" / "env_default.yaml")
    Q = np.load(q_path)
    visits = np.load(v_path)
    codes = greedy_with_unvisited(Q, visits)

    sev_labels = ["sev0(empty)", "sev1", "sev2", "sev3"]
    qlen_labels = bucket_labels(cfg.state_buckets.queue_len)
    age_labels = bucket_labels(cfg.state_buckets.oldest_age, "m")
    tleft_labels = bucket_labels(cfg.state_buckets.time_left, "m")
    crit_labels = ["c0", "c1", "c2"]

    out: list[str] = []
    out.append("# Learned policy — tabular Q-learning (run 0)\n")
    out.append("Generated by `scripts/policy_table.py` from `results/q_learning_Q.npy`.")
    out.append("Regenerate with `python scripts/train.py && python scripts/policy_table.py`.\n")
    out.append("| code | action |")
    out.append("|---|---|")
    for action, code in ACTION_CODES.items():
        out.append(f"| `{code}` | {cfg.actions.names[action]} |")
    out.append(f"| `{UNVISITED}` | **state never visited** — no decision was made here |\n")

    visited_states = sum(1 for c in codes if c != UNVISITED)
    out.append(f"**Coverage: {visited_states}/{len(codes)} states visited "
               f"({100 * visited_states / len(codes):.0f}%).** Everything marked "
               f"`{UNVISITED}` is absence of data, not a preference.\n")

    # --- view 1: the full grid, three panels
    for tleft in range(N_TLEFT):
        crunch = " — **the crunch**" if tleft == 0 else ""
        out.append(f"\n## time_left = {tleft_labels[tleft]}{crunch}\n")
        header = "| queue_len \\ oldest_age |"
        sep = "|---|"
        for sev in range(N_SEV):
            for crit in range(N_CRIT):
                header += f" {sev_labels[sev][:4]}/{crit_labels[crit]} |"
                sep += "---|"
        out.append(header)
        out.append(sep)
        for qlen in range(N_QLEN):
            for age in range(N_AGE):
                row = f"| q={qlen_labels[qlen]}, age={age_labels[age]} |"
                for sev in range(N_SEV):
                    for crit in range(N_CRIT):
                        row += f" {codes[encode(sev, qlen, age, tleft, crit)]} |"
                out.append(row)

    # --- view 2: the strategy-shift summary
    out.append("\n## Does the strategy shift as time runs out?\n")
    out.append("### View 2a — the LEARNED policy\n")
    out.append("Of the visited states in each time bucket, what fraction have each "
               "action as the greedy choice. This is what the agent would *do*, and "
               "it is the figure box 6 asks for.\n")
    out.append("| time_left | " + " | ".join(
        f"`{c}`" for c in ACTION_CODES.values()) + " | states visited |")
    out.append("|---|" + "---|" * (len(ACTION_CODES) + 1))

    greedy_by_bucket: dict[int, np.ndarray] = {}
    for tleft in range(N_TLEFT):
        counts = np.zeros(len(ACTION_CODES), dtype=np.float64)
        n_visited = 0
        for state in range(len(codes)):
            if decode(state)[3] != tleft or codes[state] == UNVISITED:
                continue
            n_visited += 1
            for action, code in ACTION_CODES.items():
                if codes[state] == code:
                    counts[action] += 1
        shares = counts / counts.sum() if counts.sum() else counts
        greedy_by_bucket[tleft] = shares
        out.append(f"| {tleft_labels[tleft]} | "
                   + " | ".join(f"{s * 100:.1f}%" for s in shares)
                   + f" | {n_visited} |")

    out.append("\n### View 2b — where the agent spent its EXPERIENCE\n")
    out.append("Action share weighted by visit count. This is the epsilon-greedy "
               "*behaviour* policy accumulated over the whole run, so it includes "
               "the early high-epsilon phase and is partly random by construction. "
               "Kept because coverage matters for reading view 1 — but it is not "
               "the learned policy, and the two must not be confused.\n")
    header = "| time_left | " + " | ".join(
        f"{cfg.actions.names[a]} (`{c}`)" for a, c in ACTION_CODES.items()
    ) + " | states visited |"
    out.append(header)
    out.append("|---|" + "---|" * (len(ACTION_CODES) + 1))

    shares_by_bucket: dict[int, np.ndarray] = {}
    for tleft in range(N_TLEFT):
        totals = np.zeros(len(ACTION_CODES), dtype=np.float64)
        n_visited = 0
        for state in range(len(codes)):
            if decode(state)[3] != tleft:
                continue
            if visits[state].sum() == 0:
                continue
            n_visited += 1
            totals += visits[state]
        shares = totals / totals.sum() if totals.sum() else totals
        shares_by_bucket[tleft] = shares
        row = f"| {tleft_labels[tleft]} | " + " | ".join(f"{s * 100:.1f}%" for s in shares)
        row += f" | {n_visited} |"
        out.append(row)

    out.append("\n> These shares describe the **behaviour policy during training** "
               "(epsilon-greedy, so partly random), not the final greedy policy. "
               "They show where the agent spent its experience. For what the "
               "learned policy actually does at evaluation time, see E-008's "
               "action breakdown.\n")

    path = ROOT / "results" / "policy_table.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")

    print(f"policy table -> {path}")
    print(f"coverage: {visited_states}/{len(codes)} states visited")
    print("\nLEARNED POLICY — action share over visited states:")
    for tleft in range(N_TLEFT):
        shares = greedy_by_bucket[tleft]
        parts = "  ".join(f"{ACTION_CODES[a]} {shares[a] * 100:5.1f}%" for a in ACTION_CODES)
        print(f"  {tleft_labels[tleft]:>9s}: {parts}")
    print("\nEXPERIENCE (behaviour policy, visit-weighted, includes exploration):")
    for tleft in range(N_TLEFT):
        shares = shares_by_bucket[tleft]
        parts = "  ".join(f"{ACTION_CODES[a]} {shares[a] * 100:5.1f}%" for a in ACTION_CODES)
        print(f"  {tleft_labels[tleft]:>9s}: {parts}")


if __name__ == "__main__":
    main()
