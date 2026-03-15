from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from datetime import datetime
from typing import Tuple
from pydantic import BaseModel

# Try multiple TTS options for best quality
TTS_ENGINE = None
USE_ML_MODEL = False

# Option 1: Try gTTS (Google TTS - High Quality, requires internet)
try:
    from gtts import gTTS
    TTS_ENGINE = "gtts"
    print("Using Google TTS (gTTS) - High Quality")
except ImportError:
    pass

# Option 2: Try pyttsx3 as fallback
if TTS_ENGINE is None:
    try:
        import pyttsx3
        TTS_ENGINE = "pyttsx3"
        print("Using pyttsx3 - Offline TTS")
    except ImportError:
        print("ERROR: No TTS engine available. Install: pip install gtts pyttsx3")

# Try to import transformers for ML-based detection
try:
    from transformers import pipeline
    USE_ML_MODEL = True
except Exception as e:
    print(f"Warning: Could not load transformers: {e}")
    print("Using rule-based emotion detection as fallback")

# -------------------------------------------
# FastAPI App Setup
# -------------------------------------------
app = FastAPI(title="Empathy Engine Enhanced")

STATIC_DIR = "static"
TEMPLATES_DIR = "templates"
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

AUDIO_PATH = os.path.join(STATIC_DIR, "voice_output.mp3")

# -------------------------------------------
# Load Emotion Detection Model
# -------------------------------------------
if USE_ML_MODEL:
    try:
        emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
            trust_remote_code=True
        )
        print("ML emotion detection model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        USE_ML_MODEL = False

# -------------------------------------------
# Emotion Classification Mapping
# -------------------------------------------
EMOTION_MAP = {
    "joy": "positive",
    "love": "positive",
    "optimism": "positive",
    "sadness": "negative",
    "anger": "negative",
    "disgust": "negative",
    "fear": "concerned",
    "surprise": "surprised",
    "neutral": "neutral"
}

# -------------------------------------------
# Voice Parameter Configuration (gTTS compatible)
# -------------------------------------------
VOICE_SETTINGS = {
    "positive": {
        "speed_factor": 1.15,  # Faster for excitement
        "lang": "en",
        "tld": "com",  # US accent
        "slow": False
    },
    "negative": {
        "speed_factor": 0.85,  # Slower for sadness
        "lang": "en",
        "tld": "co.uk",  # UK accent (slightly more formal/serious)
        "slow": False
    },
    "neutral": {
        "speed_factor": 1.0,
        "lang": "en",
        "tld": "com",
        "slow": False
    },
    "surprised": {
        "speed_factor": 1.25,  # Faster for surprise
        "lang": "en",
        "tld": "com.au",  # Australian accent (more expressive)
        "slow": False
    },
    "concerned": {
        "speed_factor": 0.9,  # Slightly slower
        "lang": "en",
        "tld": "co.in",  # Indian accent (can sound more measured)
        "slow": False
    }
}

# pyttsx3-specific settings
PYTTSX3_SETTINGS = {
    "positive": {
        "base_rate": 190,
        "base_volume": 0.95,
    },
    "negative": {
        "base_rate": 110,
        "base_volume": 0.75,
    },
    "neutral": {
        "base_rate": 150,
        "base_volume": 0.85,
    },
    "surprised": {
        "base_rate": 205,
        "base_volume": 0.95,
    },
    "concerned": {
        "base_rate": 125,
        "base_volume": 0.78,
    }
}

# -------------------------------------------
# Rule-Based Emotion Detection
# -------------------------------------------
def detect_emotion_fallback(text: str) -> Tuple[str, float]:
    """Simple rule-based emotion detection with intensity."""
    text_lower = text.lower()
    
    exclamations = text.count('!')
    questions = text.count('?')
    
    positive_words = ['happy', 'joy', 'excited', 'wonderful', 'great', 'amazing', 
                     'excellent', 'fantastic', 'love', 'best', 'perfect', 'awesome']
    positive_count = sum(1 for word in positive_words if word in text_lower)
    
    negative_words = ['sad', 'angry', 'hate', 'terrible', 'awful', 'bad', 
                     'worst', 'horrible', 'disappointed', 'frustrated']
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    surprised_words = ['wow', 'surprised', 'shocked', 'unexpected', 'unbelievable', 
                      'incredible', 'astonishing']
    surprised_count = sum(1 for word in surprised_words if word in text_lower)
    
    concerned_words = ['worried', 'concerned', 'anxious', 'afraid', 'scared', 
                      'nervous', 'uncertain', 'doubtful']
    concerned_count = sum(1 for word in concerned_words if word in text_lower)
    
    if surprised_count > 0 or (questions > 0 and exclamations > 0):
        intensity = min(0.6 + (surprised_count * 0.15) + (exclamations * 0.1), 1.0)
        return "surprised", intensity
    
    if concerned_count > 0:
        intensity = min(0.5 + (concerned_count * 0.2), 1.0)
        return "concerned", intensity
    
    if positive_count > negative_count:
        intensity = min(0.5 + (positive_count * 0.15) + (exclamations * 0.1), 1.0)
        return "positive", intensity
    
    if negative_count > positive_count:
        intensity = min(0.5 + (negative_count * 0.15) + (exclamations * 0.1), 1.0)
        return "negative", intensity
    
    return "neutral", 0.5

# -------------------------------------------
# ML-Based Emotion Detection
# -------------------------------------------
def detect_emotion_ml(text: str) -> Tuple[str, float]:
    """Detect emotion using ML model and map to our 5 categories."""
    
    results = emotion_classifier(text)

    # If nested list (older pipeline behavior)
    if isinstance(results[0], list):
        results = results[0]

    top_result = max(results, key=lambda x: x["score"])

    raw_emotion = top_result["label"].lower()
    intensity = top_result["score"]

    mapped_emotion = EMOTION_MAP.get(raw_emotion, "neutral")

    return mapped_emotion, round(intensity, 3)

# -------------------------------------------
# Main Emotion Detection Function
# -------------------------------------------
def detect_emotion(text: str) -> Tuple[str, float]:
    """Detect emotion and intensity from text."""
    if USE_ML_MODEL:
        return detect_emotion_ml(text)
    else:
        return detect_emotion_fallback(text)

# -------------------------------------------
# Generate Speech with gTTS (High Quality)
# -------------------------------------------
def generate_speech_gtts(text: str, emotion: str, intensity: float) -> dict:
    """Generate high-quality TTS using Google TTS."""
    settings = VOICE_SETTINGS.get(emotion, VOICE_SETTINGS["neutral"])
    
    # Create gTTS object with emotion-specific parameters
    tts = gTTS(
        text=text,
        lang=settings["lang"],
        tld=settings["tld"],
        slow=settings["slow"]
    )
    
    # Save the audio file
    tts.save(AUDIO_PATH)
    
    # Calculate display parameters
    speed_factor = settings["speed_factor"]
    adjusted_speed = int(150 * speed_factor)  # Base 150 wpm
    
    return {
        "rate": adjusted_speed,
        "pitch": round(0.8 + (intensity * 0.6), 2),  # Simulated pitch
        "volume": round(0.85 + (intensity * 0.15), 2),
        "accent": settings["tld"]
    }

# -------------------------------------------
# Generate Speech with pyttsx3 (Enhanced)
# -------------------------------------------
def generate_speech_pyttsx3(text: str, emotion: str, intensity: float) -> dict:
    """Generate TTS using pyttsx3 with optimizations."""
    import pyttsx3
    
    settings = PYTTSX3_SETTINGS.get(emotion, PYTTSX3_SETTINGS["neutral"])
    
    engine = pyttsx3.init()
    
    # Try to select better voices
    voices = engine.getProperty('voices')
    
    best_voice = None
    for voice in voices:
        voice_name = voice.name.lower()
        if any(keyword in voice_name for keyword in ['zira', 'david', 'hazel', 'samantha', 'enhanced']):
            best_voice = voice.id
            break
    
    if best_voice:
        engine.setProperty('voice', best_voice)
    elif len(voices) > 1:
        engine.setProperty('voice', voices[1].id)
    
    # Apply intensity scaling
    rate = int(settings["base_rate"] * (1 + intensity * 0.2))
    rate = max(80, min(rate, 250))
    
    engine.setProperty("rate", rate)
    engine.setProperty("volume", settings["base_volume"])
    
    # Change file extension to .wav for pyttsx3
    wav_path = AUDIO_PATH.replace('.mp3', '.wav')
    engine.save_to_file(text, wav_path)
    engine.runAndWait()
    engine.stop()
    
    return {
        "rate": rate,
        "pitch": round(0.8 + (intensity * 0.6), 2),
        "volume": round(settings["base_volume"], 2)
    }

# -------------------------------------------
# Main Speech Generation Router
# -------------------------------------------
def generate_speech(text: str, emotion: str, intensity: float) -> dict:
    """Generate speech using the best available TTS engine."""
    if TTS_ENGINE == "gtts":
        return generate_speech_gtts(text, emotion, intensity)
    elif TTS_ENGINE == "pyttsx3":
        return generate_speech_pyttsx3(text, emotion, intensity)
    else:
        raise Exception("No TTS engine available")

# -------------------------------------------
# Routes
# -------------------------------------------
class EmotionRequest(BaseModel):
    text: str
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render homepage."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_type": "ML Model" if USE_ML_MODEL else "Rule-Based",
        "tts_engine": TTS_ENGINE.upper() if TTS_ENGINE else "None"
    })

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, text: str = Form(...)):
    """Process text input and generate emotional speech."""
    if not text.strip():
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Please enter some text.",
            "model_type": "ML Model" if USE_ML_MODEL else "Rule-Based",
            "tts_engine": TTS_ENGINE.upper() if TTS_ENGINE else "None"
        })
    
    try:
        emotion, intensity = detect_emotion(text)
        voice_params = generate_speech(text, emotion, intensity)
        timestamp = int(datetime.now().timestamp())
        
        # Determine audio file extension
        audio_ext = "mp3" if TTS_ENGINE == "gtts" else "wav"
        audio_url = f"/static/voice_output.{audio_ext}?t={timestamp}"
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "text": text,
            "emotion": emotion.capitalize(),
            "intensity": round(intensity * 100, 1),
            "rate": voice_params["rate"],
            "pitch": voice_params["pitch"],
            "volume": voice_params["volume"],
            "audio": audio_url,
            "model_type": "ML Model" if USE_ML_MODEL else "Rule-Based",
            "tts_engine": TTS_ENGINE.upper() if TTS_ENGINE else "None"
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"An error occurred: {str(e)}",
            "model_type": "ML Model" if USE_ML_MODEL else "Rule-Based",
            "tts_engine": TTS_ENGINE.upper() if TTS_ENGINE else "None"
        })

@app.post("/api/emotion")
async def api_emotion(data: EmotionRequest):
    """JSON API endpoint for emotion detection."""
    
    text = data.text.strip()

    if not text:
        return JSONResponse({"error": "No text provided."}, status_code=400)

    try:
        emotion, intensity = detect_emotion(text)

        if TTS_ENGINE == "gtts":
            settings = VOICE_SETTINGS.get(emotion, VOICE_SETTINGS["neutral"])
            speed_factor = settings["speed_factor"]
            rate = int(150 * speed_factor)
        else:
            settings = PYTTSX3_SETTINGS.get(emotion, PYTTSX3_SETTINGS["neutral"])
            rate = int(settings["base_rate"] * (1 + intensity * 0.2))

        return JSONResponse({
            "emotion": emotion,
            "intensity": round(intensity, 3),
            "voice_parameters": {
                "rate": rate,
                "pitch": round(0.8 + (intensity * 0.6), 2),
                "volume": round(0.85 + (intensity * 0.15), 2)
            },
            "model_type": "ML Model" if USE_ML_MODEL else "Rule-Based",
            "tts_engine": TTS_ENGINE
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -------------------------------------------
# Run Server
# -------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("Starting Enhanced Empathy Engine")
    print(f"TTS Engine: {TTS_ENGINE.upper() if TTS_ENGINE else 'None'}")
    print(f"Emotion Detection: {'ML Model' if USE_ML_MODEL else 'Rule-Based'}")
    print("Access at: http://127.0.0.1:8000")
    print("="*50 + "\n")
    
    if TTS_ENGINE is None:
        print("WARNING: No TTS engine available!")
        print("Install with: pip install gtts")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)