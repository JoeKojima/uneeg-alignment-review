"""Writes small EDF+C / EDF+D test files, with the record onsets in an EDF Annotations signal.
Test helper only: synthetic data, 16-bit, one sampling rate for every signal."""
import numpy as np


def _f(v, w):
    s = str(v)
    if len(s) > w:
        s = f"{float(v):.{max(0, w - 6)}g}"
    return s[:w].ljust(w).encode("ascii")


def write_edf(path, labels, fs, segments, record_s=1.0, unit="uV", phys=(-5000.0, 5000.0), kind=None,
              onset_jitter=None, start_offset=0.0):
    """segments: list of (onset_s, data) with data shaped (n_signals, n_samples) in `unit`.
    Each segment must hold a whole number of records. onset_jitter: optional per-record offsets
    (seconds) added to the written onsets, to test the gap tolerance."""
    spr = int(round(fs * record_s))
    ann_words = 60
    records = []
    for onset, data in segments:
        data = np.asarray(data, dtype=float)
        assert data.shape[0] == len(labels) and data.shape[1] % spr == 0, "segment must hold whole records"
        for r in range(data.shape[1] // spr):
            records.append((start_offset + onset + r * record_s, data[:, r * spr:(r + 1) * spr]))
    if kind is None:
        kind = "EDF+D" if len(segments) > 1 else "EDF+C"
    pmin, pmax = phys
    dmin, dmax = -32768, 32767
    ns = len(labels) + 1
    head = b"0".ljust(8) + b"X X X X".ljust(80) + b"Startdate X X X X".ljust(80)
    head += b"01.01.85" + b"00.00.00" + _f(256 * (ns + 1), 8) + kind.encode().ljust(44)
    head += _f(len(records), 8) + _f(record_s if record_s != int(record_s) else int(record_s), 8) + _f(ns, 4)
    sig_labels = list(labels) + ["EDF Annotations"]
    nsamp = [spr] * len(labels) + [ann_words]
    head += b"".join(lab.encode().ljust(16)[:16] for lab in sig_labels)
    head += b"".join(b"".ljust(80) for _ in sig_labels)
    head += b"".join(unit.encode("latin-1").ljust(8) for _ in labels) + b"".ljust(8)
    head += b"".join(_f(pmin, 8) for _ in labels) + _f(-1, 8)
    head += b"".join(_f(pmax, 8) for _ in labels) + _f(1, 8)
    head += b"".join(_f(dmin, 8) for _ in sig_labels)
    head += b"".join(_f(dmax, 8) for _ in sig_labels)
    head += b"".join(b"".ljust(80) for _ in sig_labels)
    head += b"".join(_f(n, 8) for n in nsamp)
    head += b"".join(b"".ljust(32) for _ in sig_labels)
    assert len(head) == 256 * (ns + 1)
    gain = (pmax - pmin) / (dmax - dmin)
    with open(path, "wb") as f:
        f.write(head)
        for k, (onset, block) in enumerate(records):
            if onset_jitter is not None:
                onset += onset_jitter[k]
            dig = np.clip(np.round((block - pmin) / gain + dmin), dmin, dmax).astype("<i2")
            f.write(dig.tobytes())
            tal = f"+{onset:.6f}".rstrip("0").rstrip(".").encode() + b"\x14\x14\x00"
            f.write(tal.ljust(ann_words * 2, b"\x00"))
    return path
