import os
from huggingface_hub import hf_hub_download
import librosa
import soundfile as sf

TARGET_SR = 16000

os.makedirs("data/real", exist_ok=True)
os.makedirs("data/fake/elevenlabs", exist_ok=True)
os.makedirs("data/fake/edge_tts", exist_ok=True)

reals = ['real/yt_0000_part_001.flac', 'real/yt_0000_part_002.flac', 'real/yt_0000_part_003.flac', 'real/yt_0000_part_004.flac']
fakes = ['fake/el_0001_c_part_002.flac', 'fake/el_0001_c_part_003.flac', 'fake/el_0001_c_part_004.flac', 'fake/el_0001_c_part_005.flac']

print("Converting selection clips to data/real and data/fake/elevenlabs...")
for r in reals:
    src = hf_hub_download('garystafford/deepfake-audio-detection', r, repo_type='dataset')
    base_name = os.path.splitext(os.path.basename(r))[0] + ".wav"
    dst = os.path.join("data", "real", base_name)
    y, _ = librosa.load(src, sr=TARGET_SR, mono=True)
    sf.write(dst, y, TARGET_SR, subtype="PCM_16")
    print(f"Saved real: {dst} ({len(y)/TARGET_SR:.2f}s)")

for f in fakes:
    src = hf_hub_download('garystafford/deepfake-audio-detection', f, repo_type='dataset')
    base_name = os.path.splitext(os.path.basename(f))[0] + ".wav"
    dst = os.path.join("data", "fake", "elevenlabs", base_name)
    y, _ = librosa.load(src, sr=TARGET_SR, mono=True)
    sf.write(dst, y, TARGET_SR, subtype="PCM_16")
    print(f"Saved fake (elevenlabs): {dst} ({len(y)/TARGET_SR:.2f}s)")

# Write note tracking which clips were seen during model selection
with open("data/SELECTION_CLIPS.txt", "w") as f:
    f.write("SEEN CLIPS (Used in Task 2 to select motheecreator/Deepfake-audio-detection):\n")
    f.write("\nREAL:\n")
    for r in reals:
        f.write(f"- {os.path.splitext(os.path.basename(r))[0]}.wav\n")
    f.write("\nFAKE (ElevenLabs):\n")
    for f_item in fakes:
        f.write(f"- {os.path.splitext(os.path.basename(f_item))[0]}.wav\n")

print("Data setup complete. Ready for gen_fakes.py.")

