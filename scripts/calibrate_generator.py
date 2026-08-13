"""Phase 0 calibration check (ROADMAP.md) — measure the generator, don't guess.

Generates 100 shifts and reports the three numbers the project stands on:
  1. alerts per shift        (target ≈ 170, brief §4.1)
  2. true-incident rate      (target 2.5%–3.5%, brief §4.2)
  3. Pearson r(severity, is_true_incident)  (target band 0.30–0.40 — THE assumption)

Tuning loop: run this, adjust the TUNE values in config/env_default.yaml,
run again. When both targets pass, the final numbers go into EXPLAIN.md Part 8.

Seeds 1000–1099 are used here: disjoint from both training seeds (1–10) and
evaluation seeds (101–105), so calibration never touches what we later
train or evaluate on (CONSTRAINTS.md #2).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soc_triage.config import load_env_config
from soc_triage.generator import generate_shift

N_SHIFTS = 100
CALIBRATION_SEEDS = range(1000, 1000 + N_SHIFTS)


def main() -> None:
    cfg = load_env_config(Path(__file__).resolve().parents[1] / "config" / "env_default.yaml")

    # Pool every alert from every shift; the correlation is computed over the
    # pooled population of alerts, which is what "severity predicts truth" means.
    severities: list[int] = []
    truths: list[bool] = []
    per_shift_counts: list[int] = []
    per_shift_true: list[int] = []

    for seed in CALIBRATION_SEEDS:
        alerts = generate_shift(cfg, seed)
        per_shift_counts.append(len(alerts))
        per_shift_true.append(sum(a.is_true_incident for a in alerts))
        severities.extend(a.severity for a in alerts)
        truths.extend(a.is_true_incident for a in alerts)

    sev = np.array(severities, dtype=float)
    tru = np.array(truths, dtype=float)

    alerts_per_shift = float(np.mean(per_shift_counts))
    incident_rate = float(tru.mean())
    # np.corrcoef returns the 2x2 correlation matrix; [0, 1] is r(sev, truth).
    pearson_r = float(np.corrcoef(sev, tru)[0, 1])

    lo, hi = cfg.incident.target_severity_corr
    rate_ok = 0.025 <= incident_rate <= 0.035
    corr_ok = lo <= pearson_r <= hi

    print(f"shifts generated        : {N_SHIFTS} (seeds {CALIBRATION_SEEDS.start}..{CALIBRATION_SEEDS.stop - 1})")
    print(f"alerts per shift        : {alerts_per_shift:.1f} (target ~170)")
    print(f"total alerts pooled     : {len(sev)}")
    print(f"true-incident rate      : {incident_rate * 100:.2f}%  (target 2.5-3.5%)  {'PASS' if rate_ok else 'FAIL'}")
    print(f"true incidents per shift: {np.mean(per_shift_true):.1f} +/- {np.std(per_shift_true):.1f}")
    print(f"pearson r(sev, truth)   : {pearson_r:.3f}  (target {lo:.2f}-{hi:.2f})  {'PASS' if corr_ok else 'FAIL'}")

    # Context for tuning: how predictive is each severity level on its own?
    print("\nP(true | severity):")
    for level in cfg.severity.levels:
        mask = sev == level
        print(f"  severity {level}: {tru[mask].mean() * 100:5.2f}%   (n={int(mask.sum())})")

    if rate_ok and corr_ok:
        print("\nCALIBRATION PASSED — record these numbers in EXPLAIN.md Part 8.")
    else:
        print("\nCalibration not yet in band. Tune the TUNE values in config/env_default.yaml and rerun.")


if __name__ == "__main__":
    main()
