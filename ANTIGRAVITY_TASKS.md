# Task brief for Antigravity: finish the Voice Clone Detector prototype

## Context
We are Team CyberSync (B.Tech CSE-AIML, 2nd year). Tomorrow's project presentation needs a **working prototype** of "Explainable AI Voice-Clone Detection": it listens to a call or voice note, scores whether the voice is cloned, shows **where and why**, checks for scam phrases, and gives safe advice. We have about **2 hours** total. Priority is a **reliable live demo**, not research novelty.

## Hard rules
1. **Do not train a model from scratch.** Use a pretrained Hugging Face audio-classification model (already wired in `vcd.py`).
2. **Never fabricate results.** Every accuracy/EER number must come from `evaluate.py` run on our own clips. If data is small, label it "preliminary".
3. Keep it simple: Streamlit + the files below. No new frameworks.
4. Audio stays local; nothing is uploaded or stored.
5. The explanation is **window-level localization + a deletion test**, not Grad-CAM. Do not call it Grad-CAM in the UI.

## Files already in this folder
| File | Purpose |
|---|---|
| `vcd.py` | Core logic: load audio, phone-line simulation, sliding windows, detector wrapper, median aggregation, 3-level verdict, deletion test, acoustic cues, scam-phrase check (Whisper), fusion, advice |
| `app.py` | Streamlit UI: upload/record, verdict badge, spectrogram with flagged windows, reasons, deletion test, transcript, advice, safe-word challenge, report links |
| `evaluate.py` | Batch evaluation on `data/real` and `data/fake/<generator>` -> EER, accuracy, F1 (clean vs phone-simulated), detection rate per generator, `results.csv` |
| `requirements.txt` | Dependencies |

Tested here on synthetic audio: windowing, phone simulation, scam-phrase matching, EER maths, and the app running in dev mode (`VCD_STUB=1`). **Not tested: the real Hugging Face model**, because the build machine had no access to it. Step 2 below does that.

## Tasks in priority order

### 1. Setup (target 10 min)
- Create a venv, `pip install -r requirements.txt`. Install `ffmpeg` (needed for .m4a / WhatsApp audio).
- Run `streamlit run app.py`. Confirm the page loads and the model downloads.
- If a model download fails, check internet/proxy; `VCD_STUB=1` only gives dummy scores for UI work.

### 2. Verify the real model works (target 15 min) **most important**
- In the app, upload one clip you KNOW is real (a teammate's recording) and one KNOWN clone.
- Check the sidebar "Which model label means FAKE?" is correct (guess may be wrong if labels are like `LABEL_0`). Set the right label as the default in `app.py` once confirmed.
- Try each model in `MODELS` (`app.py`) on the same two clips plus 2-3 more, and **keep the one that separates real from fake best**. Community models vary a lot in quality. Remove models that fail to load.
- Acceptance: the real clip scores clearly lower than the clone on the chosen model.

### 3. Collect a small test set (target 25 min, split across teammates)
- `data/real/`: 10-15 clips of ~10 s from the 3 team members (Hindi/English/Hinglish mix, phone microphone). Written consent from each person.
- `data/fake/elevenlabs/`, `data/fake/xtts/` (or any other generator): 10 clips each, cloned from consenting teammates. Use a different generator folder name for each tool; the folder name is reported as the "generator".
- Run: `python evaluate.py --data data --model <chosen model> --fake-label <fake label>`
- Copy the printed CLEAN and PHONE lines into slide 11's table (mark "preliminary"; sample is small). If phone accuracy is much worse than clean, that is a **valid finding**: say it honestly and explain it as the limitation we address with phone-line augmentation as future work.

### 4. Demo polish (target 30 min)
- Pick 4 reliable demo clips and test them repeatedly: (a) real voice, (b) cloned voice, (c) the same clone with "Simulate phone line" ticked, (d) a clip containing scam language ("share the OTP, this is urgent") to show the scam-phrase check and fusion.
- Put them in `demo_clips/` and add a short list under the uploader in `app.py` so we can pick one quickly.
- Take screenshots for slide 10 (replace the mockup) and the results table.
- Make sure the UI never crashes on: very short audio, silence, mp3, m4a, a 60 s file.

### 5. Optional upgrades only if time remains (in this order)
1. Cache results per file so re-analysis is instant.
2. Speaker check: enroll a trusted voice and compare using an ECAPA-TDNN embedding (`speechbrain/spkrec-ecapa-voxceleb`) and show a match/mismatch reason. Mark as a stretch goal on the slides if not finished.
3. Hindi transcript support: switch Whisper to `base` with `language=None` and test accuracy.

### 6. Fallback plan (target 15 min before presenting)
- Screen-record a successful full demo (30-60 s) in case Wi-Fi or the laptop fails.
- Keep all models pre-downloaded (run the app once fully offline-ready) and test with Wi-Fi off.

## Things the team must keep consistent with the slides
- Slide 8 tags: say "Prototype" only for features that actually work in the app (scam-phrase check, safe-word challenge, "can't tell" state, report links). Mark speaker verification as "Stretch goal" unless step 5.2 is done.
- Slide 5/6/7 mention WavLM/AASIST fine-tuning. If we only used a pretrained community model, say so: "we evaluated pretrained wav2vec-based detectors and compared them on our own test set; fine-tuning on ASVspoof is future work."
- Slide 9: replace "Grad-CAM/SHAP" wording with "window-level localization + deletion test" if asked what the app does today.

## 2-hour schedule
| Time | Do |
|---|---|
| 0:00-0:15 | Setup, run app (task 1) |
| 0:15-0:30 | Verify model, pick best (task 2) |
| 0:30-0:55 | Record real clips and make clones in parallel (task 3) |
| 0:55-1:10 | Run `evaluate.py`, fill slide 11 |
| 1:10-1:40 | Demo polish, screenshots (task 4) |
| 1:40-2:00 | Fallback video, rehearse (task 6) |

## Honest talking points if judges ask
- "Why a pretrained model?" Training from scratch needs far more time and data than a mini project; we focus on evaluation, robustness testing, explanation and protection layers.
- "Accuracy?" Quote only `evaluate.py` output on our own clips, say the sample is small.
- "What is Grad-CAM?" It is our planned explanation method; in the prototype we localize by scoring overlapping windows and verify with a deletion test.
