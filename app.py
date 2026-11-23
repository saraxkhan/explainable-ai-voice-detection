"""Streamlit UI. Run:  streamlit run app.py
Dev mode without a model:  VCD_STUB=1 streamlit run app.py   (dummy scores, UI work only)"""
import os
import random
import tempfile
import time

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import vcd

st.set_page_config(page_title="Voice Clone Detector", page_icon="\U0001F6E1", layout="wide")

MODELS = ["MelodyMachine/Deepfake-audio-detection", "motheecreator/Deepfake-audio-detection",
          "Zeyadd-Mostaffa/Deepfake-Audio-Detection-v1"]
COLORS = {"Safe": "#0A7F75", "Suspicious": "#B07C00", "Likely Cloned": "#E0245E"}


@st.cache_resource(show_spinner="Loading detector model (first run downloads it)...")
def get_detector(model_id):
    if os.environ.get("VCD_STUB") == "1":
        return vcd.StubDetector()
    return vcd.Detector(model_id)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Settings")
model_id = st.sidebar.selectbox("Detector model", MODELS)
try:
    det = get_detector(model_id)
except Exception as e:
    st.error(f"Could not load model: {e}\n\nCheck internet for the first download, or run with VCD_STUB=1 for UI work.")
    st.stop()
if isinstance(det, vcd.StubDetector):
    st.warning("DEV MODE: dummy scores. This is NOT a real detector. Do not present these results.")
idx = det.labels.index(det.guess) if det.guess in det.labels else 0
det.fake_label = st.sidebar.selectbox("Which model label means FAKE?", det.labels, index=idx,
                                      help="Check this on one known-real and one known-fake clip.")
win_s = st.sidebar.slider("Window length (s)", 2.0, 5.0, 3.0, 0.5)
hop_s = st.sidebar.slider("Window hop (s)", 0.5, 3.0, 1.5, 0.5)
low, high = st.sidebar.slider("Verdict thresholds (Safe | Suspicious | Cloned)", 0.05, 0.95, (0.35, 0.65), 0.05)
phone = st.sidebar.checkbox("Simulate phone line (8 kHz, codec, noise)", help="Robustness demo: degrade the audio first.")
use_asr = st.sidebar.checkbox("Scam-phrase check (Whisper tiny)", value=True)

# ------------------------------------------------------------------- header
st.title("\U0001F6E1 Explainable AI Voice-Clone Detector")
st.caption("Team CyberSync  |  Detect, explain and protect. Audio is processed on this machine and not stored.")

tab_up, tab_rec = st.tabs(["Upload voice note / call", "Record"])
data, fname = None, "clip.wav"
with tab_up:
    f = st.file_uploader("Upload audio", type=["wav", "mp3", "ogg", "opus", "m4a", "flac"])
    if f:
        data, fname = f.getvalue(), f.name
with tab_rec:
    if hasattr(st, "audio_input"):
        rec = st.audio_input("Record a clip")
        if rec:
            data, fname = rec.getvalue(), "recording.wav"
    else:
        st.info("Upgrade Streamlit (>=1.39) to enable recording, or use Upload.")

if data:
    st.audio(data)
if st.button("Analyze", type="primary", disabled=data is None):
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(fname)[1] or ".wav", delete=False) as tmp:
        tmp.write(data)
    try:
        y = vcd.load_audio(tmp.name)
    except Exception as e:
        st.error(f"Could not read audio: {e}")
        st.stop()
    if phone:
        y = vcd.simulate_phone(y)
    t0 = time.time()
    res = vcd.analyze(y, det, win_s, hop_s, low, high)
    latency = time.time() - t0
    text = vcd.transcribe(y) if use_asr else ""
    hits = vcd.scam_hits(text) if text else []
    fused, level = vcd.fuse(res["agg"], hits, low, high)
    st.session_state["out"] = dict(y=y, res=res, text=text, hits=hits, fused=fused, level=level,
                                   latency=latency, cues=vcd.acoustic_cues(y))

out = st.session_state.get("out")
if out:
    y, res, hits, level = out["y"], out["res"], out["hits"], out["level"]
    st.markdown(f"<div style='padding:14px 18px;border-radius:10px;background:{COLORS[level]};color:white;"
                f"font-size:26px;font-weight:700'>{level}  ·  risk {out['fused']*100:.0f}%</div>", unsafe_allow_html=True)
    if level == "Suspicious" and low <= res["agg"] < high:
        st.caption("The model is not confident either way, so we say so instead of forcing a verdict.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Voice fake score (median)", f"{res['agg']*100:.0f}%")
    c2.metric("Scam words found", len(hits))
    c3.metric("Windows analysed", len(res["scores"]))
    c4.metric("Processing time", f"{out['latency']:.1f}s for {len(y)/vcd.SR:.0f}s audio")

    # ---- explanation plot: spectrogram with flagged windows + per-window score
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [3, 1.3]}, sharex=True)
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=vcd.SR, n_mels=64, fmax=8000), ref=np.max)
    librosa.display.specshow(S, sr=vcd.SR, x_axis="time", y_axis="mel", fmax=8000, ax=ax[0], cmap="magma")
    mids = []
    for (s, e), p in zip(res["windows"], res["scores"]):
        mids.append((s + e) / 2 / vcd.SR)
        if p >= high:
            ax[0].axvspan(s / vcd.SR, e / vcd.SR, color="#FF4D6D", alpha=0.30)
    ax[0].set_title("Spectrogram: red = windows the model flagged as likely cloned")
    ax[1].plot(mids, res["scores"], marker="o", color="#7C6CF0")
    ax[1].axhline(high, color="#E0245E", ls="--", lw=1)
    ax[1].axhline(low, color="#0A7F75", ls="--", lw=1)
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("P(fake)")
    ax[1].set_xlabel("time (s)")
    st.pyplot(fig)

    left, right = st.columns(2)
    with left:
        st.subheader("Why it was flagged")
        reasons = []
        flagged = [(s / vcd.SR, e / vcd.SR) for (s, e), p in zip(res["windows"], res["scores"]) if p >= high]
        if flagged:
            reasons.append(f"Model flagged {len(flagged)} window(s), e.g. {flagged[0][0]:.1f}–{flagged[0][1]:.1f} s")
        reasons += out["cues"]
        if hits:
            reasons.append("Scam language heard: " + ", ".join(f"“{h}”" for h in hits))
        for r in reasons or ["No strong warning signs found"]:
            st.markdown(f"- {r}")
        st.caption("Acoustic cues are simple heuristics, shown as hints only.")
        st.subheader("Explanation check (deletion test)")
        if res["del_k"]:
            st.write(f"Removing the {res['del_k']} most-flagged window(s) changes the fake score "
                     f"from **{res['del_before']*100:.0f}%** to **{res['del_after']*100:.0f}%**. "
                     "A big drop means the highlighted regions really drive the decision.")
        else:
            st.write("Clip too short for the deletion test.")
    with right:
        st.subheader("What to do")
        st.info(vcd.ADVICE[level])
        st.markdown("**Safe-word challenge**: " + random.Random(len(y)).choice(vcd.CHALLENGES))
        st.markdown("[Report on cybercrime.gov.in](https://cybercrime.gov.in)  ·  or call **1930** (national cybercrime helpline)")
        if out["text"]:
            t = out["text"]
            for h in hits:
                t = t.replace(h, f"**{h}**")
            st.subheader("Transcript")
            st.write(t)
        elif use_asr:
            st.caption("Transcript unavailable (Whisper not installed or no internet for first download).")
