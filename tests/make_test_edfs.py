"""Writes two synthetic EDF+C recordings with clinical-style labels at 256 Hz, plus a clip table."""
import csv, os, sys
import numpy as np
import pyedflib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_review_batches as mrb

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(5)
fs, dur = 256, 120
labels = ["EEG FP1-REF", "EEG F3-REF", "EEG C3-REF", "EEG P3-REF", "EEG F7-REF", "EEG T7-REF", "EEG P7-REF", "EEG O1-REF",
          "EEG FZ-REF", "EEG CZ-REF", "EEG PZ-REF", "EEG FP2-REF", "EEG F4-REF", "EEG C4-REF", "EEG P4-REF", "EEG F8-REF",
          "EEG T8-REF", "EEG P8-REF", "EEG O2-REF", "ECG EKG-REF", "PHOTIC-REF"]
rows = []
for k in range(2):
    clips = [mrb.synth_clip(rng, kind, at, 120, fs, 12, 0.0) for kind, at in [("spike", "T3"), ("none", ""), ("spike", "F8"), ("blink", "")]]
    sig = np.zeros((21, fs * dur))
    for j, c in enumerate(clips):
        start = (10 + 25 * j) * fs
        sig[:20, start:start + c.shape[1]] = c
    sig[:19] += rng.standard_normal((19, fs * dur)) * 3 + 2000.0  # DC offset as many clinical systems record
    if k == 1:
        sig[6] = 0.0  # P7 (T5) disconnected in the second recording
    path = os.path.join(out, f"rec{k + 1}.edf")
    w = pyedflib.EdfWriter(path, 21, file_type=pyedflib.FILETYPE_EDFPLUS)
    hdrs = []
    for i, lab in enumerate(labels):
        hdrs.append({"label": lab, "dimension": "uV", "sample_frequency": fs, "physical_max": 5000.0,
                     "physical_min": -5000.0, "digital_max": 32767, "digital_min": -32768, "transducer": "", "prefilter": ""})
    w.setSignalHeaders(hdrs)
    w.writeSamples([s for s in sig])
    w.close()
    for j, (kind, at) in enumerate([("spike", "T3"), ("none", ""), ("spike", "F8"), ("blink", "")]):
        rows.append({"source_path": path, "center_s": f"{10 + 25 * j + 6:.2f}", "patient": f"TEST-{k + 1}",
                     "morgoth_prob": "0.9" if kind == "spike" else "0.1", "bin": "90-100%" if kind == "spike" else "10-20%",
                     "truth": kind, "truth_max": at})
rows.append({"source_path": os.path.join(out, "rec1.edf"), "center_s": "118.00", "patient": "TEST-1",
             "morgoth_prob": "0.2", "bin": "20-30%", "truth": "none", "truth_max": ""})  # runs past the end on purpose
with open(os.path.join(out, "clips.csv"), "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wtr.writeheader(); wtr.writerows(rows)
print("wrote", out)
