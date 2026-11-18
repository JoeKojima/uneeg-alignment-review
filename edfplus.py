#!/usr/bin/env python3
"""Reads windows from EDF+D (discontinuous) files on the recording's own clock.

An EDF+D file stores data records that are not necessarily back to back in time. Each
record's start time (its onset, in seconds after the header start time) is the first
entry of its "EDF Annotations" signal. Readers that ignore the onsets, MNE among them,
glue the records together and silently shift everything after the first gap.

This reader takes the onsets, splits the file into contiguous segments (two records are
contiguous when the second starts within --tol, 1 ms by default, of where the first one
ends), and reads a window [t0, t1) given in seconds on the recording clock. A window that
touches a gap or falls outside every segment is refused rather than patched.

Any other file (EDF, EDF+C) is one segment on the clock of the file itself: 0 is its first
sample, the way pyedflib and the EMU detectors read it, whatever onset its first record
carries. Its onsets are not read at all, which keeps opening a file down to its header.
With clock="onsets" every EDF+ file is placed on its record onsets instead, EDF+C included
(its first sample then sits at its first onset, often a fraction of a second): the clock
the thesis segio reader and the UNEEG detector runs use.

Records that cannot belong to the recording are left out, the way segio leaves them out:
a record whose onset is unreadable, and a record whose onset runs backward (it starts
before the end of the record kept before it, such as a last record stamped back at 0).
`dropped` counts them.

Only record onsets and the samples a window needs are read, through a memory map, so a
24 h file costs one pass over its annotation bytes and nothing else.

    python edfplus.py segments FILE.edf [--onsets]

prints the file's segments (start, duration, gap before), with no signal values or
header identifiers.
"""
import os
import sys
from collections import OrderedDict

import numpy as np

UNIT_TO_UV = {"uv": 1.0, "µv": 1.0, "μv": 1.0, "microvolt": 1.0, "mv": 1e3, "v": 1e6, "nv": 1e-3}
GAP_TOL_S = 0.001

# record onsets already read, by file: reading them touches every record of the file (one pass over a 24 h file on
# a network disk takes many seconds), and packing a study opens the same file once per clip
_ONSETS = OrderedDict()
_ONSETS_MAX = 128


def _num(b):
    s = b.decode("ascii", "replace").strip()
    return float(s) if s else 0.0


class EDFPlus:
    """Header, record onsets and contiguous segments of one EDF or EDF+ file (16-bit only)."""

    def __init__(self, path, tol=GAP_TOL_S, clock="auto"):
        if clock not in ("auto", "onsets"):
            raise ValueError(f"clock must be auto or onsets, not {clock!r}")
        self.path = path
        self.tol = tol
        self.clock = clock
        self.dropped = 0
        with open(path, "rb") as f:
            fixed = f.read(256)
            if len(fixed) < 256:
                raise ValueError(f"{path} is too short to be an EDF file")
            if fixed[0:1] == b"\xff":
                raise ValueError(f"{path} is a BDF file; this reader handles 16-bit EDF only")
            self.header_bytes = int(_num(fixed[184:192]))
            self.kind = fixed[192:236].decode("ascii", "replace").strip()
            self.n_records = int(_num(fixed[236:244]))
            self.record_s = _num(fixed[244:252])
            ns = int(_num(fixed[252:256]))
            sig = f.read(ns * 256)

        off = 0
        widths = [("label", 16), ("transducer", 80), ("dim", 8), ("pmin", 8), ("pmax", 8),
                  ("dmin", 8), ("dmax", 8), ("prefilter", 80), ("nsamp", 8), ("reserved", 32)]
        cols = {}
        for name, w in widths:
            cols[name] = [sig[off + i * w:off + (i + 1) * w] for i in range(ns)]
            off += w * ns
        self.labels = [b.decode("ascii", "replace").strip() for b in cols["label"]]
        self.units = [b.decode("latin-1").strip() for b in cols["dim"]]
        self.pmin = np.array([_num(b) for b in cols["pmin"]])
        self.pmax = np.array([_num(b) for b in cols["pmax"]])
        self.dmin = np.array([_num(b) for b in cols["dmin"]])
        self.dmax = np.array([_num(b) for b in cols["dmax"]])
        self.nsamp = np.array([int(_num(b)) for b in cols["nsamp"]])
        if self.record_s <= 0:
            raise ValueError(f"{path} has a record duration of {self.record_s} s")
        self.record_words = int(self.nsamp.sum())
        self.word_offset = np.concatenate([[0], np.cumsum(self.nsamp)[:-1]])
        size = os.path.getsize(path)
        expect = self.header_bytes + self.n_records * self.record_words * 2
        if self.n_records < 0 or size < expect:
            # EDF allows n_records = -1 while recording; trust the file size instead
            self.n_records = (size - self.header_bytes) // (self.record_words * 2)
        self._data = np.memmap(path, dtype="<i2", mode="r", offset=self.header_bytes,
                               shape=(self.n_records, self.record_words))
        self.annot = [i for i, lab in enumerate(self.labels) if lab == "EDF Annotations"]
        if self.discontinuous or (clock == "onsets" and self.annot):
            self.onsets = self._read_onsets()
        else:
            self.onsets = np.arange(self.n_records) * self.record_s
        self.keep = self._keep()
        self.dropped = int(self.n_records - self.keep.sum())
        self.segments = self._split()

    @property
    def discontinuous(self):
        return self.kind.startswith("EDF+D")

    def _read_onsets(self):
        if not self.annot:
            raise ValueError(f"{self.path} says EDF+D but has no EDF Annotations signal")
        i = self.annot[0]
        a, n = self.word_offset[i], self.nsamp[i]
        st = os.stat(self.path)
        key = (os.path.abspath(self.path), st.st_size, st.st_mtime_ns, self.header_bytes, self.n_records,
               self.record_words, int(a), int(n))
        hit = _ONSETS.get(key)
        if hit is not None:
            _ONSETS.move_to_end(key)
            return hit.copy()
        onsets = self._parse_onsets(a, n)
        _ONSETS[key] = onsets.copy()
        while len(_ONSETS) > _ONSETS_MAX:
            _ONSETS.popitem(last=False)
        return onsets

    def _parse_onsets(self, a, n):
        raw = np.ascontiguousarray(self._data[:, a:a + n]).view(np.uint8)
        onsets = np.empty(self.n_records)
        for r in range(self.n_records):
            tal = raw[r].tobytes()
            end = len(tal)
            for stop in (b"\x14", b"\x15"):
                k = tal.find(stop)
                if 0 <= k < end:
                    end = k
            text = tal[:end].decode("ascii", "replace").strip("\x00 ")
            try:
                onsets[r] = float(text)
            except ValueError:
                onsets[r] = np.nan  # left out by _keep
        if not np.isfinite(onsets).any():
            raise ValueError(f"{self.path}: no record has a readable onset")
        return onsets

    def _keep(self):
        """Records that belong to the recording: a readable onset that does not run back before the end of the
        record kept before it."""
        keep = np.isfinite(self.onsets)
        end = -np.inf
        for r in range(self.n_records):
            if not keep[r]:
                continue
            if self.onsets[r] < end - self.tol:
                keep[r] = False
                continue
            end = self.onsets[r] + self.record_s
        return keep

    def _split(self):
        """[(first_record, n_records, start_s, end_s)] for each contiguous stretch."""
        kept = np.flatnonzero(self.keep)
        if not len(kept):
            return []
        runs, first = [], kept[0]
        for prev, r in zip(kept[:-1], kept[1:]):
            # a new stretch after a gap in time, or after a record that was left out
            if r != prev + 1 or abs(self.onsets[r] - (self.onsets[prev] + self.record_s)) > self.tol:
                runs.append((first, prev))
                first = r
        runs.append((first, kept[-1]))
        return [(int(a), int(b - a + 1), float(self.onsets[a]), float(self.onsets[b] + self.record_s)) for a, b in runs]

    @property
    def end_s(self):
        """Where the recording ends on its clock: the latest end of any stretch."""
        return max((s[3] for s in self.segments), default=0.0)

    def segment_at(self, t0, t1):
        for seg in self.segments:
            if seg[2] - 1e-9 <= t0 and t1 <= seg[3] + 1e-9:
                return seg
        near = min(self.segments, key=lambda s: min(abs(s[2] - t0), abs(s[3] - t1))) if self.segments else None
        where = f"; nearest segment runs {near[2]:.3f}-{near[3]:.3f} s" if near else ""
        raise ValueError(f"window {t0:.3f}-{t1:.3f} s in {os.path.basename(self.path)} is not inside one "
                         f"contiguous segment{where}")

    def read(self, t0, t1):
        """[(label, fs, data_in_uV, unit)] for every non-annotation signal, for [t0, t1) on the recording clock."""
        first, nrec, s0, _s1 = self.segment_at(t0, t1)
        out = []
        for i, lab in enumerate(self.labels):
            if i in self.annot:
                continue
            fs = self.nsamp[i] / self.record_s
            a = int(round((t0 - s0) * fs))
            n = int(round((t1 - t0) * fs))
            r_lo = a // self.nsamp[i]
            r_hi = min(nrec, -(-(a + n) // self.nsamp[i]))
            w0 = self.word_offset[i]
            block = np.asarray(self._data[first + r_lo:first + r_hi, w0:w0 + self.nsamp[i]], dtype=np.float64).ravel()
            k = a - r_lo * self.nsamp[i]
            dig = block[k:k + n]
            if len(dig) != n:
                raise ValueError(f"window {t0:.3f}-{t1:.3f} s runs past the end of its segment")
            gain = (self.pmax[i] - self.pmin[i]) / ((self.dmax[i] - self.dmin[i]) or 1.0)
            phys = (dig - self.dmin[i]) * gain + self.pmin[i]
            unit = self.units[i].strip().lower()
            out.append((lab, float(fs), phys * UNIT_TO_UV.get(unit, 1.0), unit))
        return out

    def close(self):
        self._data = None


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    onsets = "--onsets" in argv
    argv = [a for a in argv if a != "--onsets"]
    if len(argv) != 2 or argv[0] != "segments":
        sys.exit("usage: python edfplus.py segments FILE.edf [--onsets]")
    f = EDFPlus(argv[1], clock="onsets" if onsets else "auto")
    print(f"{os.path.basename(argv[1])}: {f.kind or 'EDF'}, {f.n_records} records of {f.record_s:g} s, "
          f"{len(f.segments)} contiguous segment(s)" + (f", {f.dropped} record(s) left out" if f.dropped else "")
          + (", on the record-onset clock" if onsets else ""))
    prev_end = None
    for k, (_a, n, s, e) in enumerate(f.segments):
        gap = "" if prev_end is None else f", gap before {s - prev_end:.3f} s"
        print(f"  segment {k}: starts {s:.3f} s, lasts {e - s:.3f} s{gap}")
        prev_end = e


if __name__ == "__main__":
    main()
