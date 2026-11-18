"""Writes a made-up study for rehearsing the live route without patient data:

  OUT/edf/EMU9901_iic_0.edf   EDF+C, 210 s
  OUT/edf/EMU9902_iic_0.edf   EDF+D, two segments (0-90 s and 150-270 s on the recording clock)
  OUT/edf/EMU9903_iic_0.edf   EDF+C, 210 s
  OUT/runs/cyclops/<clip>_scores.csv   per-second scores, conditions full19 and subq_L, in the
                                       shape a detector run writes (second, condition, max_score)
  OUT/admissions.csv          EMU9901 and EMU9903 are two admissions of one fake patient

Every 30 s of EEG holds a synthetic 12 s clip (a spike, a blink, or nothing), and the fake
scores are high at the spikes. All signals are synthetic.

    python tests/make_fake_study.py OUT
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import make_review_batches as mrb  # noqa: E402
from edf_writer import write_edf  # noqa: E402

LABELS = ["EEG FP1-REF", "EEG F3-REF", "EEG C3-REF", "EEG P3-REF", "EEG F7-REF", "EEG T7-REF", "EEG P7-REF",
          "EEG O1-REF", "EEG FZ-REF", "EEG CZ-REF", "EEG PZ-REF", "EEG FP2-REF", "EEG F4-REF", "EEG C4-REF",
          "EEG P4-REF", "EEG F8-REF", "EEG T8-REF", "EEG P8-REF", "EEG O2-REF", "ECG EKG-REF"]
FS = 256


def stretch(rng, dur, onset, events):
    """dur seconds of synthetic EEG; one 12 s synthetic clip every 30 s. Records event times on the recording clock."""
    x = mrb.synth_clip(rng, "none", "", 0.0, FS, dur, 0.0)  # continuous background and EKG
    kinds = [("spike", "T3"), ("none", ""), ("spike", "F8"), ("blink", ""), ("spike", "O1"), ("none", ""), ("spike", "C4")]
    for j, start in enumerate(range(3, dur - 12, 30)):
        kind, at = kinds[(j + len(events)) % len(kinds)]
        seed = int(rng.integers(0, 2 ** 31))
        # the event alone: the same synthetic clip with and without it, from the same seed
        ev = mrb.synth_clip(np.random.default_rng(seed), kind, at, 120.0, FS, 12, 0.0) - \
            mrb.synth_clip(np.random.default_rng(seed), "none", "", 0.0, FS, 12, 0.0)
        x[:, start * FS:(start + 12) * FS] += ev
        events.append((onset + start + 6, kind, at))
    return x


def main(out):
    rng = np.random.default_rng(21)
    edf, runs = os.path.join(out, "edf"), os.path.join(out, "runs", "cyclops")
    os.makedirs(edf, exist_ok=True)
    os.makedirs(runs, exist_ok=True)
    plan = {"EMU9901_iic_0": [(0, 210)], "EMU9902_iic_0": [(0, 90), (150, 120)], "EMU9903_iic_0": [(0, 210)]}
    truth = []
    for name, segs in plan.items():
        events, data = [], []
        for onset, dur in segs:
            data.append((onset, stretch(rng, dur, onset, events)))
        write_edf(os.path.join(edf, name + ".edf"), LABELS, FS, data)
        rows = []
        for onset, dur in segs:
            for s in range(onset, onset + dur):
                near = [k for t, k, _a in events if abs(t - (s + 0.5)) <= 0.5]
                base = float(np.clip(rng.beta(0.7, 7.0), 0, 0.99))
                full = float(rng.uniform(0.7, 0.98)) if near and near[0] == "spike" else base
                rows.append({"second": s, "condition": "full19", "max_score": round(full, 4), "t_max": round(s + rng.random(), 3)})
                rows.append({"second": s, "condition": "subq_L", "max_score": round(base * 0.5, 4), "t_max": round(s + rng.random(), 3)})
        with open(os.path.join(runs, name + "_scores.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        truth += [{"clip": name, "t_s": t, "kind": k, "at": a} for t, k, a in events]
    with open(os.path.join(out, "admissions.csv"), "w", newline="") as f:
        f.write("emu_id,patient_anon_id\nEMU9901,FAKE-A\nEMU9902,FAKE-B\nEMU9903,FAKE-A\n")
    with open(os.path.join(out, "truth.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "t_s", "kind", "at"])
        w.writeheader()
        w.writerows(truth)
    print(f"wrote {out}: 3 fake EDF clips (one EDF+D), fake detector output, fake admissions")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python tests/make_fake_study.py OUT")
    main(sys.argv[1])
