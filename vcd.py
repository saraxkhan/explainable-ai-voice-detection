"""vcd.py - core logic for the Voice Clone Detector prototype (no UI code here).

Pipeline: load audio -> (optional phone-line simulation) -> sliding windows ->
pretrained detector scores each window -> median aggregation -> 3-level verdict
-> deletion test (explanation check) + acoustic cues + scam-phrase check.
"""
import re
import numpy as np
import librosa
from scipy.signal import butter, sosfilt

SR = 16000
FAKE_WORDS = {"fake", "spoof", "spoofed", "synthetic", "deepfake", "generated", "ai", "cloned", "clone"}
REAL_WORDS = {"real", "bonafide", "genuine", "human", "authentic"}


# ---------------------------------------------------------------- audio utils
def load_audio(path, max_seconds=60):
    """Load any audio file (wav/mp3/ogg/opus/m4a; m4a needs ffmpeg) as 16 kHz mono."""
    y, _ = librosa.load(path, sr=SR, mono=True, duration=max_seconds)
    if y.size < SR // 2:
        raise ValueError("Audio is too short (need at least 0.5 s).")
    peak = float(np.max(np.abs(y)))
    return (y / peak * 0.95).astype(np.float32) if peak > 0 else y


def simulate_phone(y, snr_db=25.0, seed=0):
    """Make clean audio sound like a phone call: 8 kHz, 300-3400 Hz band, G.711 mu-law, light noise."""
    rng = np.random.default_rng(seed)
    y8 = librosa.resample(y, orig_sr=SR, target_sr=8000)
    sos = butter(4, [300, 3400], btype="band", fs=8000, output="sos")
    y8 = sosfilt(sos, y8)
    mu = 255.0
    x = np.clip(y8, -1, 1)
    c = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((c + 1) / 2 * 255) / 255 * 2 - 1
    y8 = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    p_sig = np.mean(y8 ** 2) + 1e-12
    noise = rng.normal(0, np.sqrt(p_sig / (10 ** (snr_db / 10))), size=y8.shape)
    y8 = y8 + noise
    out = librosa.resample(y8.astype(np.float32), orig_sr=8000, target_sr=SR)
    n = len(y)
    out = np.pad(out, (0, max(0, n - len(out))))[:n]
    return out.astype(np.float32)


def make_windows(n_samples, win_s=3.0, hop_s=1.5):
    win, hop = int(win_s * SR), int(hop_s * SR)
    if n_samples <= win:
        return [(0, n_samples)]
    wins = [(s, s + win) for s in range(0, n_samples - win + 1, hop)]
    if wins[-1][1] < n_samples:  # cover the tail
        wins.append((n_samples - win, n_samples))
    return wins


# ------------------------------------------------------------------ detector
class Detector:
    """Wraps a Hugging Face audio-classification model. Returns P(fake) per window."""

    def __init__(self, model_id, fake_label=None):
        from transformers import pipeline  # lazy: heavy import
        self.model_id = model_id
        self.pipe = pipeline("audio-classification", model=model_id, top_k=None)
        self.labels = list(self.pipe.model.config.id2label.values())
        self.guess = self.guess_fake_label(self.labels)
        self.fake_label = fake_label or self.guess or self.labels[-1]

    @staticmethod
    def guess_fake_label(labels):
        for l in labels:
            words = set(re.split(r"[^a-z0-9]+", l.lower()))
            if words & FAKE_WORDS and not words & REAL_WORDS:
                return l
        return None

    def p_fake(self, chunk):
        out = self.pipe({"raw": chunk.astype(np.float32), "sampling_rate": SR})
        for o in out:
            if o["label"] == self.fake_label:
                return float(o["score"])
        raise RuntimeError(f"Label {self.fake_label!r} not in model output {[o['label'] for o in out]}")

    def score_windows(self, y, windows):
        return np.array([self.p_fake(y[s:e]) for s, e in windows])


class StubDetector:
    """DEVELOPMENT ONLY: dummy scores so the UI can be built without a model. Not a real detector."""
    model_id = "STUB (dummy scores, not a real detector)"
    labels = ["real", "fake"]
    guess = "fake"
    fake_label = "fake"

    def score_windows(self, y, windows):
        rng = np.random.default_rng(len(y))
        return np.clip(rng.normal(0.5, 0.25, len(windows)), 0, 1)


# ------------------------------------------------------------------ verdicts
def level_from_score(p, low=0.35, high=0.65):
    return "Safe" if p < low else ("Likely Cloned" if p >= high else "Suspicious")


def deletion_test(scores, frac=0.25):
    """Explanation check: remove the most-flagged windows; if the fake score drops, the highlighted regions were driving the verdict."""
    scores = np.asarray(scores)
    k = max(1, int(round(len(scores) * frac)))
    if len(scores) <= k:
        return float(np.median(scores)), float(np.median(scores)), 0
    keep = np.argsort(scores)[:-k]
    return float(np.median(scores)), float(np.median(scores[keep])), k


def analyze(y, detector, win_s=3.0, hop_s=1.5, low=0.35, high=0.65):
    windows = make_windows(len(y), win_s, hop_s)
    scores = detector.score_windows(y, windows)
    agg = float(np.median(scores))
    before, after, k = deletion_test(scores)
    return {"windows": windows, "scores": scores, "agg": agg, "level": level_from_score(agg, low, high),
            "del_before": before, "del_after": after, "del_k": k}


# ------------------------------------------------- heuristic acoustic cues
def acoustic_cues(y):
    """Simple, UNCALIBRATED cues shown as hints next to the model result (not proof of cloning)."""
    cues = []
    try:
        f0 = librosa.yin(y, fmin=70, fmax=400, sr=SR, frame_length=1024)
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0][: len(f0)]
        f0 = f0[: len(rms)][rms > 0.02]
        if len(f0) > 50:
            semis = 12 * np.log2(f0 / np.median(f0))
            if np.std(semis) < 1.5:
                cues.append("Pitch is unusually steady (real speech usually varies more)")
    except Exception:
        pass
    rms_all = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
    quiet = np.mean(rms_all < 0.02)
    if quiet < 0.03:
        cues.append("Almost no natural pauses or breaths")
    return cues


# --------------------------------------------------------- scam-phrase check
SCAM_TERMS = ["otp", "your pin", "atm pin", "pin number", "cvv", "password", "urgent", "immediately", "right now", "transfer", "send money",
              "account blocked", "account will be blocked", "arrest", "police", "cbi", "customs", "parcel", "kyc",
              "upi", "gift card", "don't tell", "do not tell", "verify your", "refund", "lottery",
              "turant", "jaldi", "paise", "bhej do", "khata band", "giraftar"]


def scam_hits(text):
    t = " " + re.sub(r"\s+", " ", text.lower()) + " "
    return sorted({w for w in SCAM_TERMS if re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", t)})


_whisper = None


def transcribe(y, size="tiny"):
    """Speech-to-text with faster-whisper (CPU). Returns '' if unavailable."""
    global _whisper
    try:
        if _whisper is None:
            from faster_whisper import WhisperModel
            _whisper = WhisperModel(size, device="cpu", compute_type="int8")
        segs, _ = _whisper.transcribe(y.astype(np.float32), beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segs)
    except Exception as e:  # no internet / not installed
        return ""


def fuse(agg, hits, low=0.35, high=0.65):
    """Combine voice score and scam language into one risk level + advice."""
    fused = min(1.0, agg + 0.10 * min(len(hits), 3))
    level = level_from_score(fused, low, high)
    if hits and level == "Safe":
        level = "Suspicious"  # human-sounding voice but scam language
    return fused, level


ADVICE = {
    "Safe": "No strong sign of a cloned voice. Still never share OTPs or PINs on a call.",
    "Suspicious": "Be careful. Do not share OTPs or send money. Hang up and call back on a number you already know.",
    "Likely Cloned": "Do not share OTPs or send money. Hang up and call the person back on a known number, then report it.",
}
CHALLENGES = ["Ask: what did we eat the last time we met?", "Ask for the family safe word you agreed earlier.",
              "Ask a question only the real person would know, such as a pet's nickname.",
              "Say you will call back and use the number saved in your contacts."]
