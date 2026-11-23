# EEG annotation GUI [vShare]

A shareable copy of the IED Clip Review kit (version 1.8.6). It holds the code, the rater manuals and one synthetic demo batch, and no study data: no EEG, clip tables, keys, labels, rater links or site credentials.

**Start here**
- Open `ied_review.html` in a browser. It runs offline on the synthetic demo batch in `sample_batch/`.
- Python needs numpy, scipy and pandas, plus pyedflib for EDF input and pyarrow for parquet tables. Run the tests with `python tests/test_tools.py`, `python tests/test_phase5.py`, `python tests/test_hosted.py` and `node tests/test_core.js`.
- To host a study of your own, follow `hosted/site/README.md`. You need your own private GitHub repository, Vercel project and Upstash Redis database. The site repository holds packed EEG, so keep it private and separate from this one.
- The server was written for a shared Linux cluster reached over SSH. `serve_review.py --ssh-host` sets the host name it prints in the tunnel command.
- `.gitignore` keeps EEG files, clip tables, keys, labels and tokens out of this repository. Keep it that way.


Tools for collecting blinded expert labels on 10-second scalp EEG clips: an adapter that reads a detector's per-clip output (MORGOTH or Cyclops), a sampler that draws weighted clips from the per-second scores, a server on Borel that reads each clip straight from the EEG files when a rater opens it, a hosted site (a private GitHub repository served by Vercel) that gives each rater a link, a viewer clinicians use in their browser (from the site, through an SSH tunnel to Borel, or from disk with batch files), and scripts that join the labels back to the sources and report the results.


| File | What it does |
|---|---|
| `ied_review.html` | The viewer. Opens in Chrome, Edge, Firefox or Safari straight from disk and works offline. It starts on a synthetic demo batch. |
| `select_clips.py` | `from-runs` turns a detector's output folder into one row per scored second (columns found automatically, times moved onto each EDF's clock and checked against its length, EMU IDs mapped to patients); `inspect` shows the folder's column layout without values; `from-edf` lists every second of plain EDF files when no detector has run yet; `check-data` reads a sample of the real clips the way the server does and prints only aggregates; `join` puts several scorings of the same seconds side by side (raw and SD-matched, or several detectors), keeps only seconds inside a pool of intervals (`--pool`, carrying each second's SubQ file and clock) and leaves out flagged seconds (`--drop-flagged`, plus `--drop-unflagged` for seconds the flag table has no row for; one table is enough with these; seconds match to the nearest row within `--match-tol-s`, default 0.001 s); `sample` draws a stratified, weighted clip table (`--prob-cols` bins each second by the highest of several scorings; `--draw` makes several draws, each binned on its own score, and merges them; `--random` adds clips drawn with no score; `--grid-s 12` keeps one candidate second per 12 s of each file so clips never share EEG and the weights stay exact; `--disjoint` keeps later draws off the seconds earlier ones picked); `admin` writes the owner-only admin table (SubQ, implant side, every detector's score for each second of each clip); `fake-morgoth` writes made-up probabilities for test EDFs. |
| `serve_review.py` | Serves the viewer on Borel's localhost with a token. With `--clips` it reads each clip straight from the EEG files (live mode, no pack step); given a folder it serves packed batch files. `--raters EC,JK` gives each rater a link of their own. Labels save back to Borel as raters work; `--status` prints progress. |
| `report_labels.py` | Turns `merged_labels.csv` into a one-page HTML report: weighted IED share with a patient-level interval; for every joined scoring, weighted precision, sensitivity and F1 at a threshold (`--threshold`, 0.5 by default), F1 across thresholds and calibration against the expert consensus; rater agreement (pairs, Fleiss' kappa, Gwet's AC1, each rater against the others); per-patient shares; the field-maximum head map and test-retest only when the labels have them. `demo` simulates a kit 1.6 study (four scorings, four raters, shared and split clips) with a known truth. |
| `hosted/` | `make_site.py` writes the site folder (blinded batches, the viewer, link hashes) from the same study and labels folders the server uses; `check_upload.py` refuses anything that could tie a clip to a patient (it is also the site's pre-commit hook); `pull_labels.py` copies the site's labels back to Borel. `hosted/site/` is the site's template: one Node function, no dependencies. |
| `make_review_batches.py` | `pack` writes batch files from a clip table; `synthetic` writes test batches with planted spikes. |
| `merge_labels.py` | Joins exported label CSVs with `key.csv` and prints summaries, including agreement over all raters (Fleiss' kappa and Gwet's AC1). |
| `edfplus.py` | Reads windows from EDF+D (discontinuous) files on the recording's own clock, refusing any window that touches a gap. `python edfplus.py segments FILE.edf` lists a file's segments. |
| `sample_batch/` | One synthetic batch (20 clips) and its key, for trying the viewer without patient data. |
| `src/`, `build.py`, `tests/` | Viewer source, the build step that produces `ied_review.html`, and tests. |

Python needs numpy, scipy and pandas; `pyedflib` for EDF input; `mne` only for `--reader mne`.

## Time-alignment review (branch `alignment-review`)

Reviewers check by eye that the UNEEG device is on scalp time, on the seizures that were annotated on both
systems (the scalp-EEG consensus and the UNEEG reviewers' consensus, converted to scalp time by the lab, within
60 s of each other): one clip per seizure, from 30 s before to 30 s after the scalp onset, paged 10 s (or 5 s)
at a time. The 10-20 EEG is on top, the virtual UNEEG pairs nearest the implant under it (F3-T3 and F7-T3 on the
left, F4-T4 and F8-T4 on the right) and the device's own channels at the bottom (D-C, C-P and D-P); the scalp
onset is marked red and the UNEEG onset green. For each clip they answer **Time aligned**, **Not time aligned**
or **Unsure** (keys Y, N, U) and can add a note; nothing moves on until they press the right arrow or Next.

- Build the batch on the server, from the continuous analysis' EDF+D files (one per patient, device already on
  scalp time):
  ```
  python make_alignment_review.py --data /users/joekoji/uneeg_continuous \
      --sides /users/joekoji/uneeg_continuous/continuous_patient_info.csv \\
      --onsets /users/joekoji/montage-proj/uneeg_seizure_figures/seizure_onsets.csv --out align_review
  ```
  It writes `align_review/align01.json.gz` (the clips) and `align01_key.csv` (patient, seizure, onsets, implant
  side). Both hold patient data: share them privately, never through this repository. `--random` makes one
  random 60-s clip per patient instead.
- Hosted with a password: `python encrypt_batch.py align_review/align01.json.gz reviews/align01.enc --new-password-file <outside the repo>/PASSWORD.txt` encrypts the batch (AES-256-GCM, key from PBKDF2-SHA256 of the password); `ied_review.html?review=align01` (where `index.html` points) asks for the password, decrypts in the browser and opens the review, starting each clip centred on its scalp onset with a 20-s page. Scroll sideways (or Shift + scroll) to move in time, up or down for amplitude. The .enc file is public; only the password protects it, so share the password separately.
- Review: open `ied_review.html` (or the hosted page), choose **Open batch**, pick `align01.json.gz`, enter your
  initials. Keys: `[` `]` (or Page Up / Page Down, or the arrows in the Time control) page through the clip,
  B / T / A switch to double banana, transverse or average reference (M still cycles; A is not the artifact flag
  here), Y / N / U answer, right arrow next patient. **Manual** in the top bar explains the review; it opens by
  itself the first time a browser opens the review (remembered in the browser's storage). **Export labels** saves
  the CSV to send back; `decision` is yes (time aligned), no (not time aligned) or unsure, with the note.
- Try it without data: `python make_alignment_review.py --synthetic --out demo_alignment` writes three made-up
  clips with shared blinks and chewing; the third has the device 1.5 s late, so it should be marked Not time
  aligned. The same demo batch is in `demo_alignment/`.
- Viewer changes (src/core.js, src/page.html; rebuild with `python build.py`): batches with
  `"question": "alignment"` take their question and answer wording from `question_text` and `answer_labels`,
  page through the whole clip from its start, and hide the field maximum, artifact and mark controls (G and D do
  nothing there) and the clip details.

## Workflow

0. **Read the detector's output** (skip if you already have one row per scored second).
   ```
   python select_clips.py inspect out/cyclops_emu
   python select_clips.py from-runs out/cyclops_emu --condition full19 --detector cyclops \
       --clip-root /path/to/edf/clips --admissions data_derived/emu_admissions.csv --out seconds.csv
   ```
   `inspect` prints each table layout in the folder (column names, types, row counts, the conditions present, and which columns it would use), never values. `from-runs` reads every table under the folder: one per clip (the clip is the file name, or the nearest folder name with an EMU ID) or long tables with a clip column. It finds the time, score, condition, clip and path columns by name (override with `--time-col` and friends); tables without a time and score column, such as per-patient summaries, are skipped and counted. Times that end in .5, or a column named like a centre, are window centres and are moved to the start of their second (`--time-is` overrides). With several conditions in the tables it asks for `--condition`. EDF paths come from a path column (relative paths are read under `--clip-root`, so two files with the same name in different folders stay apart) or from the file names under `--clip-root`. Every time is checked against its EDF's length, so seconds counted in samples are caught. `--edf-clock` says which clock the times are on: `samples` (the default; 0 is a file's first sample and EDF+D files follow their record onsets, as the EMU runs read them) or `onsets` (every EDF+ file on its record onsets, EDF+C included, as the thesis segio reader and the UNEEG runs read them). The table records the clock in an `edf_clock` column, `join` refuses tables on different clocks, and the server reads each clip on its row's clock. Records stamped backward (a last record stamped back at 0) or with no readable onset are left out of the recording, as segio leaves them out; `python edfplus.py segments FILE.edf --onsets` shows a file's stretches and how many records were left out. `--admissions` maps each EMU ID (one admission) to its patient, so a patient with two admissions counts once; admissions missing from that table are left out and counted. It refuses any file named like an id_map.
   No detector output yet? `python select_clips.py from-edf --clip-root /path/to/edf/clips --clip-list 7f_clips.csv --admissions data_derived/emu_admissions.csv --out seconds.csv` lists every whole second of the clips (EDF+D gaps skipped) with no score, and `sample` then draws a simple random sample within each patient. Before anything else, `python select_clips.py check-data --clip-root ... --n 20` reads a 12 s window from 20 of the clips exactly as the server will and ends with OK or PROBLEM.
1. **Draw the clips** (optional; any table with the two columns below works).
   ```
   python select_clips.py sample morgoth_seconds.csv --out clips.csv --n 1000 --pilot 25
   ```
   Reads one row per scored second (`patient, source_path, start_s, prob`; rename with `--patient-col` and friends), drops seconds too close to the edge of their unbroken stretch of scored seconds (so a clip never crosses the end of a file or an EDF+D gap), splits candidates into probability bins (deciles by default), shares each bin's clips across patients so none dominates, and samples without replacement. `sampling_weight` is each cell's candidate count over its clip count, and the script stops if the weights in any bin x patient cell fail to add back up to that cell's candidates. `--pilot 25` also writes `clips_pilot.csv`, spread across bins and patients. `--grid-s 12` keeps clips from one file apart so one spike can't appear twice: only one second in every 12 s of each file (at an offset drawn per file) is a candidate, so every second keeps an equal chance and the weights stay exact. (`--min-gap-s 12` does the same by skipping seconds near earlier picks, but then a second inside a burst has less chance than its cell's share, and estimates on bursty data run a few percent low.) To judge several scorings against one set of labels, join them first and stratify on all of them: `python select_clips.py join --out seconds.csv cyclops=raw.csv cyclops_sd=sd.csv`, then `sample seconds.csv --prob-cols cyclops_prob,cyclops_sd_prob ...`. Each keeps its own column in the clip table and key.csv, the weights stay valid for each, and `sample` prints how many clips each one calls high.
   **Several draws (1.7).** `sample seconds.csv --draw spikenet2_prob=1000 --draw morgoth_prob=1000 --random 200` makes each draw on its own, binned on its own score (seed `[--seed, k]` for the k-th draw), and merges them on recording and second: a clip two draws pick is shown once, `draws` lists every draw that picked it, `w_<draw>` is that draw's own weight (blank if the draw didn't pick it; these add up exactly to the candidates in each of its bins) and `sampling_weight` is one over the chance that any draw picked the clip. `<name>_clipmax` is each scoring's highest score over the 10 s a rater sees (`--pad-s`). `clips_design.csv` keeps the design: every draw's bin x patient cells with candidates `N`, clips `n`, `p_patient` and each second's selection chance `p_second`. A pool table (`join --pool`) carries `subq_path`, `subq_start_s`, `subq_rate` and `implant_side` to each second, and `sample` adds `subq_center_s`, the drawn second's middle on the SubQ file's clock.
2. **Or make a clip table yourself.** One row per clip, with `source_path` (an EEG file) and `center_s` (center of the candidate second, in seconds from the start of that file, or on the recording clock for EDF+D). Add any other columns you want to keep with the clip: patient, absolute time, `morgoth_prob`, `bin`, `sampling_weight`. They go into `key.csv`, never into rater batches.
3. **Serve it live** (the usual route).
   ```
   python serve_review.py --clips clips.csv --study review_study --labels review_labels --raters EC,JK --shared-clips 200
   ```
   On first start this reads every clip once to prove it can be read (unreadable ones are listed and left out), fixes the clip IDs, order, repeats and batches, and writes `plan.json` and `key.csv` to the study folder (mode 600). Later starts reuse the plan, so b03 is always the same clips, and refuse to run if `clips.csv` has changed. When a rater opens a batch, the server reads its clips from the EEG files, so no copy of the EEG is written anywhere. See *Serving from Borel* below.
   A new study offers Yes and No only (`--answers ynu` adds Unsure). `--shared-clips 200` puts 200 clips in shared batches (s01, s02, ...) that every rater labels; the rest are split evenly between the raters named when links are first made, and the split is kept in `<labels>/.assignment.json`. `--reassign` deals out again the split batches nobody has started.
   **Admin and superadmin views (1.7).** `python select_clips.py admin clips.csv --out admin.csv --scalp spikenet2=sn2_seconds.csv --scalp morgoth=m_seconds.csv --subq cyclops=cyclops_subq_seconds.parquet` maps each detector's per-second scores (scalp, and SubQ through `subq_center_s` and `subq_rate`) onto each clip. Give it to `serve_review.py` or `make_site.py` as `--admin admin.csv`: your own link then gets S (virtual UNEEG lanes made from scalp bipolar pairs on the implant side, `--virtual L=F7-T3 --virtual R=F8-T4` by default, then the real SubQ channels, plus from 1.7.6 the distant pair when they are two bipolar channels sharing a contact (UNEEG D-C and P-C give D-P = (D-C) - (P-C)), the implant side and the draw that picked the clip) and D (one strip per detector with a box for every second). Rater links never receive it, except those named with `--admin-raters` (an advisor's first look; remembered in the labels folder, `none` clears it). Every answer records `view` (blind, admin or superadmin), and the report leaves out answers that were not blind.
3b. **Or host it** for raters without a Borel login.
   ```
   python hosted/make_site.py --clips clips.csv --study review_study --labels review_labels --out site --raters EC,JK --batch-size 25 --shared-clips 200
   ```
   Plans the study exactly as step 3 would (same plan, same links) and writes the site folder: blinded batches under `content/batches/`, `content/site.json` (batch list, assignment, SHA-256 hashes of the links), the viewer and the server function. It runs `check_upload.py` on the folder and fails if anything else is in it. Push the folder to a private GitHub repository that Vercel serves, with an Upstash Redis database for the answers (the site's README says how). `python hosted/pull_labels.py --labels review_labels` brings the answers back as the same `labels_<batch>_<RATER>.csv` files. Hosted batches must be under 4 MB, hence `--batch-size 25`. Add `--results-login SJ` once (it asks for a password of 12 characters or more and keeps only its scrypt hash in `review_labels/.results.json`) and every later run also writes the results page, served at `<site>/results` behind that username and password: the report below, built from key.csv and every labels file pulled so far, with a panel of what has come in. Patients show as P01, P02, ...; the key to those names is `review_labels/results_patients.csv` and stays on Borel. `--expect-scorings morgoth,cyclops,...` names the scorings the page waits for (a missing one shows as not in yet), `--extra-scores name=seconds.csv` adds a scoring that arrives after the draw (matched on recording and second), and `--threshold` sets the flagging score (0.5).
4. **Or pack it** into batch files, for raters without a Borel login.
   ```
   python make_review_batches.py pack clips.csv --out review_batches --mode rater --repeat-frac 0.05
   ```
   This writes `b01.json.gz`, `b02.json.gz`, ... (100 clips each by default), `key.csv` and `manifest.json`, and re-reads every batch to confirm the stored signal matches what was read. Rows that cannot be read are listed at the end; the rest are still packed.
   Packed batches can be served the same way (`python serve_review.py review_batches --labels review_labels`) or sent as files with `ied_review.html`. `key.csv` stays on the cluster either way.
5. **Raters label.** Pick a batch (served) or choose Open batch (files). A rater link signs the rater in; otherwise they enter initials. Labels save in the browser as they go, and when served also to `review_labels/labels_<batch>_<RATER>.csv` on Borel. With files, Export labels saves a CSV to send back. Each rater sees the clips in their own seeded random order.
6. **Merge.**
   ```
   python merge_labels.py --key review_study/key.csv review_labels/labels_b01_EC.csv review_labels/labels_b01_SJ.csv
   ```
   Writes `merged_labels.csv` (one row per clip and rater) and prints counts per rater, the Yes rate per bin, test-retest agreement on repeated clips, agreement between each pair of raters and over all raters (Fleiss' kappa and Gwet's AC1, with 95% intervals from resampling clips), and a weighted prevalence when `key.csv` has `sampling_weight`.

Try the whole loop on synthetic data first: `python make_review_batches.py synthetic --out demo_batches`, label one batch, export, and merge. The merge prints a comparison against the planted spikes.

7. **Report.**
   ```
   python report_labels.py merged_labels.csv --out label_report.html
   ```
   One HTML file that opens offline and holds aggregates only (counts, rates, patient study IDs). `python report_labels.py demo --out demo.html` shows it on a simulated 1,000-clip study.

## Serving from Borel

`serve_review.py` listens on 127.0.0.1 only, and because Borel is shared, every request must also carry a token: yours (kept in `<labels>/.token`, mode 600, so links survive a restart) or a rater's. `--raters EC,JK` gives each rater their own link (tokens in `<labels>/.raters.json`, mode 600). A rater link opens the viewer signed in as that rater, can read and save only that rater's labels, and shows only their own progress; adding raters later keeps existing links, `--revoke EC` turns one off, `--new-token` changes every link, and `--links` prints them again. It serves the viewer and the batches and nothing else: never `key.csv`, the plan, the clip table or file paths. In live mode a clip that fails to read is reported in your terminal and the browser only hears that a clip could not be read. It refuses research batch files unless `--allow-research` (live mode makes blinded batches only), checks every label upload against the batch (clip IDs, rater, columns) and writes it atomically, keeping a copy in `history/` at most every 10 minutes. Request logs never include the token.

In the viewer, Batches lists what is on the server with each rater's progress; a chip in the top bar says when labels were last saved to the server and turns red when a save fails (it retries, and the browser keeps its own copy meanwhile). Opening the same batch in another browser merges the two copies clip by clip, latest edit winning. Export still works.

Live mode needs numpy, scipy and pyedflib, like `pack`; serving batch files needs only the standard library. Raters need a Borel login to open the tunnel. Anyone without one uses the file route.

## Modes

**rater** (default): clips are shuffled, carry random IDs, and contain only waveforms. The viewer also hides any subcutaneous channel group in a rater batch.

**research**: clips keep table order and show the table's columns as clip details. Two optional columns add more: `markers`, a JSON list such as `[{"t": 6.02, "kind": "cyclops", "label": "Cyclops 0.91"}]` with `t` in seconds from the start of the 12 s clip, and `subq_path` with `subq_center_s`, which draws subcutaneous channels under the scalp traces. `subq_center_s` must already be on the scalp clock, so this waits on the alignment work.

## Readers and units

`--reader edf` (default) reads EDF and BDF with pyedflib and converts physical units to microvolts. EDF+D files, which have gaps, go through `edfplus.py`: each data record is placed at its own onset (from the EDF Annotations signal), records within 1 ms of back to back form one segment, `center_s` is read on the recording clock, and a window that touches a gap is refused rather than glued together (MNE glues them, which shifts everything after the first gap). Discontinuous BDF is refused. `--reader mne` reads anything `mne.io.read_raw` opens. To read another format, add a class with `read(path, t0, t1)` returning `[(label, fs, data_in_uV, unit), ...]` and register it in `READERS`.

Channel labels are normalized to classic 10-20 names: `EEG FP1-REF` becomes Fp1, T7 becomes T3, P8 becomes T6, and anything containing EKG or ECG becomes EKG. Missing or flat electrodes are marked bad and drawn as dashed lines. The packager prints the median scalp amplitude and warns when it is implausible, which is how a volts and microvolts mix-up shows itself.

Clips are resampled to `--fs` (200 Hz by default), each channel's mean is removed, and samples are stored as int16 with a per-channel scale. The round-trip error is at most half a quantization step.

## The viewer

Display follows clinical conventions: negative up, double banana by default with transverse and average reference available, 7 µV/mm, LFF 1 Hz, HFF 70 Hz, 60 Hz notch on when a batch opens, 10 s per page, ruled every 0.2 s with heavier lines each second. Filters are zero-phase (forward and backward) so they do not shift spike timing. EKG is scaled to fit its own row. Settings are remembered per browser.

From 1.7 a new study asks about the whole clip ("Epileptiform discharge anywhere in this clip?", `--question clip`): the page is fixed at 10 s and no second is highlighted. Studies planned before 1.7 keep `--question second`. key.csv records each clip's `question`, and the report judges detectors on `<name>_clipmax` only for clips asked about as a whole.

Keyboard in a blinded Yes/No batch (every study from 1.6 on): Y and N answer and both move on; B, T and A switch to double banana, transverse and average reference; up and down arrows step the sensitivity and left and right arrows go to the previous or next clip without answering (1.7.5); on your own link with an admin table, S and D turn the admin and superadmin views on and off; no other key does anything, and there is no field maximum, artifact flag or other-discharge marking (1.6.3). Research batches and studies that offer Unsure keep the full viewer: in a batch that offers Unsure, Y, N, U answer and only N moves on; clicking electrodes on the head map marks the field maximum and sets Yes; G generalized; A artifact; D then click to mark other discharges; Enter or right arrow next; Shift with right arrow next unlabeled; up and down arrows change sensitivity; M montage; W 10 s or 5 s page; ? help.

### Exported columns

`batch_id, clip_id, rater, order_index, decision` (yes, no, unsure or blank), `max_electrodes` (semicolon list), `side` and `region` (derived from the electrodes), `generalized, artifact` (0 or 1), `other_discharges_s` (seconds from the center of the highlighted second), `note, seconds_viewed, labeled_at, order_seed, viewer_version`, `view` (blind, admin or superadmin: what was on screen when the answer was given; from 1.7, and older files without it still load).

### Batch file format (version 1)

JSON, optionally gzipped. Header fields: `format` ("ied-review-batch"), `version`, `batch_id`, `mode`, `fs`, `clip_s`, `pad_s`, `event_s`, `candidate_s`, `channels` (scalp names, then extras), `groups` (extra channels by group, such as `{"EKG": ["EKG"]}`), `order_seed`, `answers` (`["yes", "no"]`, or with `"unsure"`; batches without it offer all three), `question` (`"clip"` or `"second"`; batches without it ask about the second), `display` (`{"notch": true}`). The owner-only admin file (`"format": "ied-review-admin"`) has `batch_id`, `fs`, `clip_s` and per clip `id`, `implant_side`, `drawn_by`, `strips` (`name`, `kind` scalp or subq, `len`, `secs` as [seconds from clip start, score]) and `subq` (`channels`, `scale`, `data`, encoded like the scalp). Each clip: `id`, `event_s`, `scale` (µV per step, one per channel), `data` (base64 of little-endian int16, channel-major), optional `bad`, and in research mode optional `meta` and `markers`.

## Tests

```
node tests/test_core.js                                  # montage, filter, encoding and CSV logic
python tests/test_tools.py                               # adapter, sampler, EDF+D reader, live mode, rater links, Yes/No, shared batches, agreement, server safety
python tests/test_hosted.py                              # site folder contents, upload check, the site's server and labels database (needs node), pull_labels
python tests/test_phase5.py                               # several draws merged and unbiased, the SubQ pool and flags, admin table, owner-only admin view, clip-level report
python tests/e2e_hosted.py /tmp/e2e_hosted /tmp/shots    # synthetic batches -> make_site -> the site served locally -> rater links label -> pull -> merge
python tests/e2e_live.py /tmp/e2e_live /tmp/shots        # fake run output -> from-runs -> sample -> live serve -> two raters label -> merge -> report
python tests/e2e_server.py /tmp/e2e /tmp/shots           # fake EDFs -> sample -> pack -> serve -> label in a browser -> merge
python tests/make_fake_study.py /tmp/fake                # fake EDF clips (one EDF+D), fake detector output, fake admissions table
python tests/make_test_edfs.py /tmp/edf                  # EDF files with clinical labels, a gap channel, a DC offset
python make_review_batches.py pack /tmp/edf/clips.csv --out /tmp/packed
python tests/e2e.py demo_batches /tmp/shots              # browser run: open, label, export, reload, merge (needs playwright)
```

The browser tests take `IED_CHROMIUM=/path/to/chrome` when Playwright's own browser isn't installed.

After editing `src/core.js` or `src/page.html`, run `python build.py` to regenerate `ied_review.html`.
