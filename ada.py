import os
import time
import threading
import cv2
import numpy as np
import pyttsx3
import whisper
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ultralytics import YOLO
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import pvporcupine

# 🔥 KI-Modell laden (Mistral-7B oder Phi-3)
model_name = "microsoft/phi-3-mini-4k-instruct"  # Klein, schnell & kostenlos
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# 🎤 Whisper für Spracherkennung
whisper_model = whisper.load_model("base")  # "small", "medium" für bessere Qualität

# 📷 YOLOv8 für Bilderkennung (Outfit-Analyse)
yolo_model = YOLO("yolov8n.pt")  # Nano-Version (schnell)

# 🔊 Text-to-Speech (kostenlos)
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # Deutsch einstellen

# 🎙️ Wake Word Detection ("Hey ADA")
porcupine = pvporcupine.create(
    access_key="kostenloser-key-für-porcupine",  # Hol dir einen auf picovoice.ai
    keyword_paths=["hey-ada_de_windows_v2_1_0.ppn"]  # Wake-Word-Modell
)

# 🎤 Offline-Spracherkennung (Vosk als Fallback)
vosk_model = Model(lang="de-de")
recognizer = KaldiRecognizer(vosk_model, 16000)

def speak(text):
    print(f"🤖 A.D.A.: {text}")
    engine.say(text)
    engine.runAndWait()

def listen(timeout=5):
    print("🎤 Höre zu...")
    start_time = time.time()
    audio_data = sd.rec(int(44100 * timeout), samplerate=44100, channels=1, dtype='float32')
    sd.wait()
    
    try:
        result = whisper_model.transcribe(audio_data.flatten(), fp16=False, language="de")
        command = result["text"]
        print(f"👤 Du: {command}")
        return command
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return ""

def analyze_outfit():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Kamera konnte nicht geöffnet werden.")
        return
    
    ret, frame = cap.read()
    if ret:
        results = yolo_model(frame)
        detected_objects = []
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = yolo_model.names[class_id]
                detected_objects.append(label)
        
        speak(f"Ich sehe: {', '.join(detected_objects)}. Vielleicht kombinierst du es mit einer passenden Hose?")
    
    cap.release()

def get_ai_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    speak("Hallo! Ich bin A.D.A., deine persönliche KI. Sage 'Hey A.D.A.' um mich zu aktivieren.")
    
    while True:
        # 🔥 Wake Word Erkennung (Hey A.D.A.)
        audio = sd.rec(int(44100 * 3), samplerate=44100, channels=1, dtype='float32')
        sd.wait()
        keyword_index = porcupine.process(audio.flatten())
        
        if keyword_index >= 0:
            speak("Ja, wie kann ich dir helfen?")
            command = listen()
            
            if "was soll ich anziehen" in command.lower():
                analyze_outfit()
            elif "zauber" in command.lower():
                speak("✨ Abrakadabra! Ich zaubere dir einen perfekten Tag!")
            else:
                response = get_ai_response(command)
                speak(response)

if __name__ == "__main__":
    main()