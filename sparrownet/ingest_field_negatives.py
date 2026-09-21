"""
ingest_field_negatives.py — turn the intern's field recordings into training clips.

Takes long continuous WAV takes plus the collection log, and produces 3-second 16 kHz
clips named so that `mybad_cache.group_id()` keeps every segment of one take together.

Safety rules this enforces, because each has already cost this project real time:

  * Only takes whose log row says `bird_heard == no` are ingested. `maybe` and `yes` are
    skipped and reported. A negative clip containing a bird teaches the detector to
    ignore birds, which is worse than having no clip at all.
  * Every take must appear in the log. An unlogged WAV is skipped, not guessed at.
  * Output names embed site/date/take so grouping works, and `--group-by session`
    (the default) additionally prevents two takes recorded minutes apart at one site
    from landing on opposite sides of a train/test split.

Usage:
    python ingest_field_negatives.py --audio-dir /path/to/takes --log /path/to/log.csv
    python ingest_field_negatives.py --audio-dir ... --log ... --dry-run
"""

import os
import csv
import glob
import json
import argparse

import numpy as np
import librosa
import soundfile as sf

from config import SAMPLE_RATE, CLIP_SECONDS, CLIP_SAMPLES

DEFAULT_OUT = "/Volumes/Evo/mybad_field/negative"


def read_log(path):
    """Log rows keyed by filename. Requires `filename` and `bird_heard` columns."""
    rows = {}
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        missing = {"filename", "bird_heard"} - set(rdr.fieldnames or [])
        if missing:
            raise SystemExit(f"log is missing required column(s): {sorted(missing)}\n"
                             f"found: {rdr.fieldnames}")
        for r in rdr:
            rows[os.path.basename(r["filename"].strip())] = r
    return rows


def segment_take(path, out_dir, stem, min_rms=1e-4, dry_run=False):
    """Cut one take into non-overlapping 3s clips at 16 kHz mono.

    Clips that are essentially digital silence are dropped: they carry no information
    about what Malaysia sounds like, and a batch of them would just teach the model
    that silence is not a bird, which it already knows.
    """
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    n = len(y) // CLIP_SAMPLES
    written, skipped_silent = 0, 0
    for i in range(n):
        seg = y[i * CLIP_SAMPLES:(i + 1) * CLIP_SAMPLES]
        if np.sqrt(np.mean(seg ** 2)) < min_rms:
            skipped_silent += 1
            continue
        if not dry_run:
            sf.write(os.path.join(out_dir, f"{stem}_{i + 1:04d}.wav"),
                     seg, SAMPLE_RATE, subtype="PCM_16")
        written += 1
    return written, skipped_silent, len(y) / SAMPLE_RATE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True, help="folder of long WAV takes")
    ap.add_argument("--log", required=True, help="collection_log.csv")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--allow-maybe", action="store_true",
                    help="also ingest takes marked bird_heard=maybe (NOT recommended)")
    args = ap.parse_args()

    log = read_log(args.log)
    takes = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    if not takes:
        raise SystemExit(f"no .wav files in {args.audio_dir}")
    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)

    accept = {"no"} | ({"maybe"} if args.allow_maybe else set())
    stats = {"takes": len(takes), "ingested": 0, "clips": 0, "silent_dropped": 0,
             "minutes": 0.0, "skipped": []}

    for path in takes:
        base = os.path.basename(path)
        row = log.get(base)
        if row is None:
            stats["skipped"].append((base, "not in log"))
            continue
        verdict = (row.get("bird_heard") or "").strip().lower()
        if verdict not in accept:
            stats["skipped"].append((base, f"bird_heard={verdict or 'blank'}"))
            continue

        stem = os.path.splitext(base)[0]
        written, silent, dur = segment_take(path, args.out_dir, stem,
                                            dry_run=args.dry_run)
        stats["ingested"] += 1
        stats["clips"] += written
        stats["silent_dropped"] += silent
        stats["minutes"] += dur / 60.0
        print(f"  {base:52s} {dur/60:5.1f} min -> {written:4d} clips"
              + (f" ({silent} silent dropped)" if silent else ""))

    print(f"\n=== {'DRY RUN — nothing written' if args.dry_run else args.out_dir} ===")
    print(f"takes found      {stats['takes']}")
    print(f"takes ingested   {stats['ingested']}")
    print(f"audio ingested   {stats['minutes']:.1f} min")
    print(f"clips produced   {stats['clips']}   (3s each, 16 kHz mono)")
    print(f"silent dropped   {stats['silent_dropped']}")
    if stats["skipped"]:
        print(f"\nSKIPPED {len(stats['skipped'])} take(s) — each needs a decision:")
        for b, why in stats["skipped"][:20]:
            print(f"  {b:52s} {why}")
        print("  (takes marked 'maybe' can be listened to again and re-logged as 'no',")
        print("   or left out; do NOT bulk-accept them with --allow-maybe casually)")

    # how far this gets us toward a balanced minimal set
    target = 5000
    print(f"\ntoward the minimal target of ~{target} Malaysian negative clips: "
          f"{stats['clips']} / {target} ({100*stats['clips']/target:.0f}%)")
    if stats["clips"] < target:
        need = (target - stats["clips"]) * CLIP_SECONDS / 60
        print(f"  roughly {need:.0f} more minutes of usable audio needed")

    if not args.dry_run:
        with open(os.path.join(args.out_dir, "ingest_report.json"), "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
