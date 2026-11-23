"""gen_fakes.py - Generate synthetic audio clips using edge-tts.

Creates 10 clips (5 English, 5 Hindi), each ~8-12 seconds, converted to 16 kHz mono WAV.
Saved in data/fake/edge_tts/.
Includes 2 scam-language clips for evaluation and testing.
"""
import asyncio
import os
import tempfile
import edge_tts
import librosa
import soundfile as sf

TARGET_DIR = os.path.join("data", "fake", "edge_tts")
TARGET_SR = 16000

CLIPS = [
    # English clips
    {
        "filename": "edge_en_01_colloquial.wav",
        "voice": "en-US-JennyNeural",
        "text": "Hey everyone, welcome back to the channel. Today we are diving into our machine learning project architecture and discussing how neural networks process complex audio signals.",
        "is_scam": False
    },
    {
        "filename": "edge_en_02_meeting.wav",
        "voice": "en-US-GuyNeural",
        "text": "Good afternoon team. I have reviewed the sprint progress and everything looks on track for tomorrow's demo. Please ensure all documentation and slide decks are updated.",
        "is_scam": False
    },
    {
        "filename": "edge_en_03_travel.wav",
        "voice": "en-IN-NeerjaNeural",
        "text": "The weather forecast for northern India predicts clear skies through the weekend, making it an ideal time for travel and outdoor campus activities across Delhi and Punjab.",
        "is_scam": False
    },
    {
        "filename": "edge_en_04_scam.wav",
        "voice": "en-GB-SoniaNeural",
        "text": "Urgent security alert from your bank! An unauthorized debit has occurred. Please share your OTP immediately right now, otherwise your account will be blocked and frozen.",
        "is_scam": True
    },
    {
        "filename": "edge_en_05_academic.wav",
        "voice": "en-IN-PrabhatNeural",
        "text": "In today's computer science seminar, we explored speech synthesis and generative models. Deep learning has drastically accelerated voice cloning fidelity over the past two years.",
        "is_scam": False
    },
    # Hindi clips
    {
        "filename": "edge_hi_01_notes.wav",
        "voice": "hi-IN-SwaraNeural",
        "text": "नमस्ते दोस्तों, कल की परीक्षा के लिए सभी महत्वपूर्ण नोट्स मैंने क्लास के व्हाट्सएप ग्रुप में साझा कर दिए हैं। कृपया एक बार सभी लोग डाउनलोड करके दोहरा लें।",
        "is_scam": False
    },
    {
        "filename": "edge_hi_02_fest.wav",
        "voice": "hi-IN-MadhurNeural",
        "text": "कॉलेज के वार्षिक तकनीकी महोत्सव में आप सभी का स्वागत है। विभिन्न तकनीकी प्रतियोगिताओं के लिए पंजीकरण काउंटर पर शुरू हो चुके हैं, जल्दी शामिल हों।",
        "is_scam": False
    },
    {
        "filename": "edge_hi_03_assignment.wav",
        "voice": "hi-IN-SwaraNeural",
        "text": "क्या आपने अपना प्रोजेक्ट असाइनमेंट पूरा कर लिया है? याद रखें कि शाम पांच बजे से पहले हमें रिपोर्ट कॉलेज पोर्टल पर अपलोड करनी होगी ताकि कोई अंक न कटे।",
        "is_scam": False
    },
    {
        "filename": "edge_hi_04_scam.wav",
        "voice": "hi-IN-MadhurNeural",
        "text": "अति आवश्यक सूचना! आपका बैंक खाता तुरंत ब्लॉक कर दिया जाएगा। अपना ओटीपी और पासवर्ड किसी को न बताएं लेकिन सत्यापन के लिए तुरंत शेयर करें, पैसे भेज दो।",
        "is_scam": True
    },
    {
        "filename": "edge_hi_05_weather.wav",
        "voice": "hi-IN-SwaraNeural",
        "text": "मौसम विभाग के ताजा बुलेटिन के अनुसार, आने वाले चौबीस घंटों में तेज हवाओं के साथ बारिश होने की संभावना है। सभी लोग घर लौटते समय सावधानी बरतें।",
        "is_scam": False
    }
]


async def generate_clip(item):
    out_path = os.path.join(TARGET_DIR, item["filename"])
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_name = tmp.name

    try:
        communicate = edge_tts.Communicate(item["text"], item["voice"])
        await communicate.save(tmp_name)
        
        # Load and convert to 16 kHz mono WAV
        y, _ = librosa.load(tmp_name, sr=TARGET_SR, mono=True)
        # normalize peak
        peak = float(librosa.util.normalize(y).max())
        if peak > 0:
            y = y / max(abs(y.min()), abs(y.max())) * 0.95
        sf.write(out_path, y, TARGET_SR, subtype="PCM_16")
        duration = len(y) / TARGET_SR
        print(f"Generated: {item['filename']} | Voice: {item['voice']} | Duration: {duration:.2f}s | Scam: {item['is_scam']}")
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


async def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Generating 10 AI speech clips into {TARGET_DIR}...")
    for item in CLIPS:
        await generate_clip(item)
    print("All edge-tts clips generated successfully!")


if __name__ == "__main__":
    asyncio.run(main())

