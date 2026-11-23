"""Batch evaluation -> REAL numbers for slide 11.

Folder layout:
  data/real/*.wav               genuine recordings (teammates, with consent)
  data/fake/<generator>/*.wav   clones, one sub-folder per generator (elevenlabs, xtts, rvc ...)

Run:  python evaluate.py --data data --model MelodyMachine/Deepfake-audio-detection --fake-label fake
Prints EER / accuracy / F1 for CLEAN and PHONE-simulated audio, plus detection rate per generator.
Saves results.csv. Small data = say "preliminary" on the slide."""
import argparse
import csv
import glob
import os

import numpy as np

import vcd


def eer(y_true, scores):
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    best = (9, 0.5)
    for t in np.concatenate([[0], np.unique(scores), [1.01]]):
        pred = scores >= t
        far = np.mean(pred[y_true == 0])      # real flagged as fake
        frr = np.mean(~pred[y_true == 1])     # fake missed
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]


def metrics(y_true, scores, thr=0.5):
    y_true, pred = np.asarray(y_true), np.asarray(scores) >= thr
    tp, fp, fn = np.sum(pred & (y_true == 1)), np.sum(pred & (y_true == 0)), np.sum(~pred & (y_true == 1))
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"acc": float(np.mean(pred == y_true)), "f1": float(2 * prec * rec / max(prec + rec, 1e-9)), "eer": float(eer(y_true, scores))}


def files(pattern_dir):
    exts = ("wav", "mp3", "flac", "ogg", "opus", "m4a")
    return [p for e in exts for p in glob.glob(os.path.join(pattern_dir, f"**/*.{e}"), recursive=True)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="motheecreator/Deepfake-audio-detection")
    ap.add_argument("--fake-label", default="fake")
    ap.add_argument("--max-seconds", type=int, default=20)
    a = ap.parse_args()

    det = vcd.Detector(a.model, a.fake_label)
    print("Model labels:", det.labels, "| using FAKE label:", det.fake_label)
    items = [(p, 0, "real") for p in files(os.path.join(a.data, "real"))]
    for g in sorted(os.listdir(os.path.join(a.data, "fake"))):
        items += [(p, 1, g) for p in files(os.path.join(a.data, "fake", g))]
    print(f"{sum(1 for i in items if i[1]==0)} real, {sum(1 for i in items if i[1]==1)} fake files")

    rows = []
    for p, label, group in items:
        y = vcd.load_audio(p, a.max_seconds)
        for cond, yy in (("clean", y), ("phone", vcd.simulate_phone(y))):
            r = vcd.analyze(yy, det)
            rows.append({"file": p, "label": label, "group": group, "cond": cond, "score": r["agg"]})
        print(".", end="", flush=True)
    print()
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    for cond in ("clean", "phone"):
        sub = [r for r in rows if r["cond"] == cond]
        m = metrics([r["label"] for r in sub], [r["score"] for r in sub])
        print(f"\n[{cond.upper()}] EER {m['eer']*100:.1f}%  accuracy {m['acc']*100:.1f}%  F1 {m['f1']*100:.1f}%")
        for g in sorted({r["group"] for r in sub if r["label"] == 1}):
            sc = [r["score"] for r in sub if r["group"] == g]
            print(f"   detection rate on {g}: {np.mean(np.array(sc) >= 0.5)*100:.0f}% ({len(sc)} clips)")
        real = [r["score"] for r in sub if r["label"] == 0]
        print(f"   real clips wrongly flagged: {np.mean(np.array(real) >= 0.5)*100:.0f}% ({len(real)} clips)")
