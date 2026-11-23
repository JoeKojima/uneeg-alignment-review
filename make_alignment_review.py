#!/usr/bin/env python3
"""
make_alignment_review.py
========================
Writes one review batch for checking the UNEEG-to-scalp time alignment by eye (alignment-review branch).

One clip per seizure that was annotated on both systems: the scalp-EEG consensus and the UNEEG reviewers'
consensus (converted to scalp time by the lab) within 60 s of each other, as listed in the seizure figure's
seizure_onsets.csv. Each clip runs from 30 s before to 30 s after the scalp onset (--before / --after), read
from the continuous analysis' EDF+D file (19 scalp channels and the device's D-C and P-C, already placed on
scalp time); time where a system was not recording shows flat. The scalp onset is marked red and the UNEEG
onset green. Clips come in patient order. The viewer pages through each clip 10 s (or 5 s) at a time and
shows, top to bottom:

    10-20 EEG (double banana, or transverse / average)
    Virtual UNEEG   scalp pairs nearest the implant: F3-T3 and F7-T3 (left), F4-T4 and F8-T4 (right)
    UNEEG           the device as recorded: D-C, C-P (= -(P-C)) and D-P (= D-C minus P-C)

With --random instead: one clip of --clip_s seconds per patient at a random time (seed 0) in an unbroken piece.

The reviewer answers Time aligned / Not time aligned / Unsure for each patient and can add a note.
Answers export as decision yes / no / unsure (yes = time aligned).

    python make_alignment_review.py --data /users/joekoji/uneeg_continuous \\
        --sides /users/joekoji/uneeg_continuous/continuous_patient_info.csv \\
        --onsets /users/joekoji/montage-proj/uneeg_seizure_figures/seizure_onsets.csv --out align_review
    python make_alignment_review.py --synthetic --out demo            # no patient data: a demo batch

Output: <out>/align01.json.gz (open it in ied_review.html with Open batch) and <out>/align01_key.csv
(each clip's patient, seizure and times; keep it with the data, not with the viewer).
"""
import argparse
import csv
import glob
import os
import sys
from datetime import datetime, timedelta

import numpy as np
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from edfplus import EDFPlus  # noqa: E402
from make_review_batches import FORMAT, FORMAT_VERSION, encode, write_batch  # noqa: E402

SCALP = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz',
         'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
VIRTUAL = {'left': [('F3', 'T3'), ('F7', 'T3')], 'right': [('F4', 'T4'), ('F8', 'T4')]}
VIRTUAL_NAMES = ['vUNEEG F3/4-T3/4', 'vUNEEG F7/8-T3/4']     # the implant side's pair
QUESTION = 'Is the UNEEG time-aligned with the 10-20 EEG?'
LABELS = {'yes': 'Time aligned', 'no': 'Not time aligned', 'unsure': 'Unsure'}


def header_start(path):
    with open(path, 'rb') as f:
        h = f.read(256)
    d, t = h[168:176].decode().strip(), h[176:184].decode().strip()
    dd, mm, yy = (int(x) for x in d.split('.'))
    HH, MM, SS = (int(x) for x in t.split('.'))
    return datetime(2000 + yy if yy < 85 else 1900 + yy, mm, dd, HH, MM, SS)


def clip_channels(sig, side, fs_out):
    """(names, rows in µV at fs_out, groups) from [(label, fs, data, unit)] of the EDF+D window."""
    by = {}
    fs_in = None
    for lab, fs, x, _u in sig:
        name = lab.replace('EEG ', '').strip()
        by[name] = x
        fs_in = fs
    up, down = int(round(fs_out)), int(round(fs_in))
    g = np.gcd(up, down)
    rs = lambda x: resample_poly(x, up // g, down // g)                    # noqa: E731
    rows, names = [], []
    for n in SCALP:
        rows.append(rs(by[n]) if n in by else None)
        names.append(n)
    # one channel name for every clip whatever the implant side (a batch has one channel list); the clip's
    # own pairs are in its details
    virt = [(nm, rs(by[a] - by[b])) for nm, (a, b) in zip(VIRTUAL_NAMES, VIRTUAL[side])]
    dc, pc = by.get('SQDC'), by.get('SQPC')
    real = [('UNEEG D-C', rs(dc)), ('UNEEG C-P', rs(-pc)), ('UNEEG D-P', rs(dc - pc))]
    n = len(rs(by[SCALP[0]]))
    bad = [nm for nm, r in zip(names, rows) if r is None]
    rows = [r if r is not None else np.zeros(n) for r in rows]
    for nm, r in virt + real:
        names.append(nm)
        rows.append(r)
    groups = {'Virtual UNEEG': [nm for nm, _ in virt], 'UNEEG': [nm for nm, _ in real]}
    return names, np.vstack(rows), groups, bad


def read_with_gaps(e, r0, r1):
    """[(label, fs, data, unit)] for [r0, r1) on the recording clock; samples where nothing was recorded are 0."""
    out = None
    for _first, _n, s0, s1 in e.segments:
        lo, hi = max(r0, s0), min(r1, s1)
        if hi - lo < 0.01:
            continue
        part = e.read(lo, hi)
        if out is None:
            out = [(lab, fs, np.zeros(int(round((r1 - r0) * fs))), u) for lab, fs, _x, u in part]
        for (lab, fs, x, _u), (_l, _f, buf, _u2) in zip(part, out):
            i0 = int(round((lo - r0) * fs))
            m = min(len(x), len(buf) - i0)
            buf[i0:i0 + m] = x[:m]
    if out is None:
        raise ValueError('nothing recorded in the window')
    return out


def pick_window(pieces, clip_s, rng):
    """A random start inside one piece long enough for the clip (pieces weighted by usable length)."""
    ok = [(p, (p['seconds'] - clip_s - 2)) for p in pieces if p['seconds'] >= clip_s + 2]
    if not ok:
        return None
    w = np.array([u for _, u in ok], float)
    p, u = ok[rng.choice(len(ok), p=w / w.sum())]
    return p, datetime.fromisoformat(p['start']) + timedelta(seconds=1 + float(rng.uniform(0, u)))


def real_batch(args):
    sides = {r['patient_id']: (r.get('implant_side') or '').strip().lower()
             for r in csv.DictReader(open(args.sides, encoding='utf-8-sig'))}
    rng = np.random.default_rng(args.seed)
    clips, key = [], []
    for d in sorted(glob.glob(os.path.join(args.data, 'US*'))):
        subj = os.path.basename(d)
        edf = os.path.join(d, f'{subj}_continuous.edf')
        pf = os.path.join(d, 'pieces.csv')
        if not (os.path.exists(edf) and os.path.exists(pf)):
            continue
        pieces = list(csv.DictReader(open(pf)))
        for p in pieces:
            p['seconds'] = float(p['seconds'])
        got = pick_window(pieces, args.clip_s, rng)
        if got is None:
            print(f'{subj}: no piece of {args.clip_s:.0f} s; skipped')
            continue
        piece, t0 = got
        side = 'left' if sides.get(subj, '').startswith('l') else 'right'
        e = EDFPlus(edf)
        rec0 = (t0 - header_start(edf)).total_seconds()
        names, rows, groups, bad = clip_channels(e.read(rec0, rec0 + args.clip_s), side, args.fs)
        e.close()
        rows = rows[:, :int(round(args.clip_s * args.fs))]
        b64, scale, _ = encode(rows)
        clips.append({'id': subj, 'event_s': 0, 'scale': scale, 'data': b64, 'bad': bad,
                      'meta': {'Patient': subj, 'Start (scalp clock, date-shifted)': t0.strftime('%Y-%m-%d %H:%M:%S'),
                               'Piece': piece['piece_id'], 'Implant side': side,
                               'Virtual UNEEG': ', '.join(a + '-' + b for a, b in VIRTUAL[side])}})
        key.append({'clip_id': subj, 'patient': subj, 'start_scalp_clock': t0.isoformat(), 'piece': piece['piece_id'],
                    'implant_side': side, 'clip_s': args.clip_s})
        print(f'{subj}: {t0:%Y-%m-%d %H:%M:%S}, {piece["piece_id"]}, implant {side}', flush=True)
    return names, groups, clips, key


def seizure_batch(args):
    """One clip per seizure annotated on both systems (seizure_onsets.csv of the seizure figure)."""
    sides = {r['patient_id']: (r.get('implant_side') or '').strip().lower()
             for r in csv.DictReader(open(args.sides, encoding='utf-8-sig'))}
    marked = lambda r: r['uneeg_onset_minus_scalp_onset_s'] not in ('', 'nan')  # noqa: E731
    # --allow_unmarked keeps seizures the UNEEG reviewers did not mark (no green line): the reviewer compares the
    # traces themselves, so a clip can check the alignment without a UNEEG mark
    rows = [r for r in csv.DictReader(open(args.onsets)) if marked(r) or args.allow_unmarked]
    rows.sort(key=lambda r: (r['subject'], int(r['sz'])))
    clips, key, names, groups = [], [], None, None
    for r in rows:
        subj, sz = r['subject'], int(r['sz'])
        on = datetime.fromisoformat(r['scalp_onset'])
        du = float(r['uneeg_onset_minus_scalp_onset_s']) if marked(r) else None
        edf = os.path.join(args.data, subj, f'{subj}_continuous.edf')
        side = 'left' if sides.get(subj, '').startswith('l') else 'right'
        e = EDFPlus(edf)
        r0 = (on - header_start(edf)).total_seconds() - args.before
        names, rows_, groups, bad = clip_channels(read_with_gaps(e, r0, r0 + args.before + args.after), side, args.fs)
        e.close()
        rows_ = rows_[:, :int(round((args.before + args.after) * args.fs))]
        b64, scale, _ = encode(rows_)
        cid = f'{subj} sz{sz}'
        dur = float(r['seizure_s'])
        markers = [{'t': args.before, 'kind': 'scalp', 'label': 'scalp onset'}]
        if du is not None:
            markers.append({'t': round(args.before + du, 3), 'kind': 'uneeg', 'label': 'UNEEG onset'})
        clips.append({'id': cid, 'event_s': 0, 'scale': scale, 'data': b64, 'bad': bad,
                      'markers': markers,
                      'meta': {'Patient': subj, 'Seizure': sz, 'Scalp onset (date-shifted)': on.strftime('%Y-%m-%d %H:%M:%S'),
                               'Seizure length (scalp)': f'{dur:.0f} s',
                               'UNEEG onset minus scalp onset': f'{du:+.1f} s' if du is not None else 'not marked on UNEEG',
                               'Implant side': side,
                               'Virtual UNEEG': ', '.join(a + '-' + b for a, b in VIRTUAL[side]),
                               'Clip': f'{args.before:.0f} s before to {args.after:.0f} s after the scalp onset'}})
        key.append({'clip_id': cid, 'patient': subj, 'seizure': sz, 'scalp_onset': on.isoformat(),
                    'uneeg_onset_minus_scalp_onset_s': du if du is not None else '', 'implant_side': side,
                    'clip_start_scalp_clock': (on - timedelta(seconds=args.before)).isoformat()})
        print(f'{cid}: scalp onset {on:%Y-%m-%d %H:%M:%S}, UNEEG onset '
              + (f'{du:+.1f} s' if du is not None else 'not marked'), flush=True)
    return names, groups, clips, key


def synthetic_batch(args):
    """Made-up EEG with shared blinks and chewing on scalp and device: for trying the viewer without data."""
    rng = np.random.default_rng(args.seed)
    n = int(round(args.clip_s * args.fs))
    t = np.arange(n) / args.fs
    clips, key = [], []
    for k in range(3):
        side = 'left' if k % 2 == 0 else 'right'
        blinks = rng.uniform(1, args.clip_s - 1, int(args.clip_s / 4))
        event = sum(np.exp(-((t - b) / 0.15) ** 2) for b in blinks) * 120
        chew = np.zeros(n)
        for c0 in rng.uniform(0, args.clip_s - 5, 2):
            m = (t > c0) & (t < c0 + 3)
            chew[m] = 40 * np.sign(np.sin(2 * np.pi * 1.5 * t[m])) * rng.standard_normal(m.sum()).clip(-2, 2)
        bg = lambda a: a * np.cumsum(rng.standard_normal(n)) / np.sqrt(args.fs) + 8 * rng.standard_normal(n)  # noqa: E731
        sc = {e: bg(6) + (event if e.startswith('Fp') or e in ('F7', 'F8', 'F3', 'F4') else 0) * (1.0 if e.startswith('Fp') else 0.4)
              + (chew if e in ('F7', 'T3', 'F8', 'T4', 'T5', 'T6') else 0) for e in SCALP}
        shift = 0.0 if k < 2 else 1.5                                   # the third demo clip is 1.5 s off
        sh = lambda x: np.interp(t - shift, t, x)                         # noqa: E731
        dc = bg(4) + sh(event) * 0.5 + sh(chew) * 0.8
        pc = bg(4) + sh(chew) * 0.3
        sig = [(e, args.fs, sc[e], 'uv') for e in SCALP] + [('SQDC', args.fs, dc, 'uv'), ('SQPC', args.fs, pc, 'uv')]
        names, rows, groups, bad = clip_channels(sig, side, args.fs)
        b64, scale, _ = encode(rows)
        cid = f'demo{k + 1}'
        clips.append({'id': cid, 'event_s': 0, 'scale': scale, 'data': b64,
                      'meta': {'Patient': f'Demo {k + 1} (synthetic)', 'Implant side': side,
                               'Virtual UNEEG': ', '.join(a + '-' + b for a, b in VIRTUAL[side])}})
        key.append({'clip_id': cid, 'patient': f'demo{k + 1}', 'implant_side': side, 'true_shift_s': shift})
    return names, groups, clips, key


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', help='continuous analysis folder: <patient>/<patient>_continuous.edf and pieces.csv')
    ap.add_argument('--sides', help='patient info csv with patient_id and implant_side')
    ap.add_argument('--out', required=True)
    ap.add_argument('--onsets', help='seizure_onsets.csv of the seizure figure (plot_seizure_alignment.py)')
    ap.add_argument('--before', type=float, default=30.0, help='seconds shown before the scalp onset')
    ap.add_argument('--after', type=float, default=30.0, help='seconds shown after the scalp onset')
    ap.add_argument('--random', action='store_true', help='one random clip per patient instead of the seizures')
    ap.add_argument('--clip_s', type=float, default=60.0, help='clip length with --random or --synthetic')
    ap.add_argument('--fs', type=float, default=200.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch_id', default='align01')
    ap.add_argument('--allow_unmarked', action='store_true',
                    help='also make clips of seizures in --onsets without a UNEEG onset (no green line)')
    ap.add_argument('--synthetic', action='store_true', help='a demo batch from made-up EEG (no patient data)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names, groups, clips, key = (synthetic_batch(args) if args.synthetic else real_batch(args) if args.random
                                 else seizure_batch(args))
    if not args.synthetic and not args.random:
        args.clip_s = args.before + args.after
    obj = {'format': FORMAT, 'version': FORMAT_VERSION, 'batch_id': args.batch_id, 'mode': 'research',
           'synthetic': bool(args.synthetic), 'created': datetime.now().isoformat(timespec='seconds'),
           'fs': args.fs, 'clip_s': args.clip_s, 'pad_s': 0, 'event_s': 0, 'candidate_s': 1,
           'channels': names, 'groups': groups, 'order_seed': 0, 'dc_removed': True, 'display': {'notch': True},
           'answers': ['yes', 'no', 'unsure'], 'question': 'alignment', 'question_text': QUESTION,
           'answer_labels': LABELS,
           'notes': 'Time-alignment review: scalp EEG, virtual UNEEG and UNEEG on scalp time.',
           'clips': clips}
    path = os.path.join(args.out, args.batch_id + '.json.gz')
    write_batch(path, obj, True)
    with open(os.path.join(args.out, args.batch_id + '_key.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    print(f'wrote {path}: {len(clips)} clips of {args.clip_s:.0f} s')


if __name__ == '__main__':
    main()
