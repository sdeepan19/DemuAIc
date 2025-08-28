# import webview
# import random
# import base64
# import numpy as np
# from transformers import pipeline
# import spotipy
# from spotipy.oauth2 import SpotifyOAuth
# import cv2
# from PIL import Image
# import speech_recognition as sr
# import webbrowser

# # Spotify Authentication
# sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
#     client_id="8016d9d06cde4ed8a397f52d3c41dd99",
#     client_secret="9aa2f8b033104ec0b4092a685a6714b0",
#     redirect_uri="http://127.0.0.1:8888/callback",
#     scope="user-read-playback-state,user-modify-playback-state,"
#           "user-read-currently-playing,user-library-read,user-top-read,playlist-read-private"
# ))

# # NLP & Face Pipelines
# text_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
# face_classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

# # Mood map
# mood_map = {
#     "joy": "happy upbeat pop",
#     "sadness": "emotional sad acoustic",
#     "anger": "heavy rock metal",
#     "fear": "calm piano instrumental",
#     "neutral": "chill lofi hip hop",
#     "surprise": "energetic edm",
#     "disgust": "dark experimental",
# }

# # Face Detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # Speech Recognition
# recognizer = sr.Recognizer()

# class DemuAicAPI:
#     """Python API for frontend."""

#     def analyze_text(self, text):
#         result = text_classifier(text)[0]
#         mood = result["label"].lower()
#         return self.get_track_for_mood(mood, confidence=result["score"])

#     def analyze_face(self, image_data):
#         img_bytes = base64.b64decode(image_data.split(",")[1])
#         np_arr = np.frombuffer(img_bytes, np.uint8)
#         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#         if frame is None:
#             return {"error": "Could not decode image"}

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(60, 60))
#         if len(faces) == 0:
#             return {"error": "No face detected"}

#         (x, y, w, h) = faces[0]
#         face = frame[y:y+h, x:x+w]
#         face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#         face_pil = Image.fromarray(face_rgb)
#         preds = face_classifier(face_pil)
#         mood = preds[0]["label"].lower()
#         return self.get_track_for_mood(mood)

#     def analyze_voice(self):
#         try:
#             with sr.Microphone() as source:
#                 recognizer.adjust_for_ambient_noise(source)
#                 audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
#                 text = recognizer.recognize_google(audio)
#             return self.analyze_text(text)
#         except Exception as e:
#             return {"error": f"Voice recognition failed: {str(e)}"}

#     def get_track_for_mood(self, mood, confidence=None):
#         query = mood_map.get(mood, "chill music")
#         results = sp.search(q=query, type="track", limit=5)["tracks"]["items"]
#         if not results:
#             return {"error": "No track found for mood"}

#         chosen = random.choice(results)
#         song_name = chosen["name"]
#         artist = chosen["artists"][0]["name"]
#         url = chosen["external_urls"]["spotify"]

#         # Try to play on Spotify Premium; otherwise, open in browser
#         try:
#             devices = sp.devices()
#             if devices["devices"]:
#                 device_id = devices["devices"][0]["id"]
#                 sp.start_playback(device_id=device_id, uris=[chosen["uri"]])
#         except spotipy.exceptions.SpotifyException:
#             # If playback fails (e.g., Free account), open in web browser
#             webbrowser.open(url)

#         return {
#             "mood": mood,
#             "confidence": round(confidence, 2) if confidence else None,
#             "song": f"{song_name} - {artist}",
#             "url": url
#         }

# if __name__ == "__main__":
#     api = DemuAicAPI()
#     window = webview.create_window("DemuAic - AI Mood Music", "home.html", js_api=api)
#     webview.start(debug=True)
import webview
import random
import base64
import numpy as np
from transformers import pipeline
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import cv2
from PIL import Image
import speech_recognition as sr
import webbrowser

# Spotify Authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="8016d9d06cde4ed8a397f52d3c41dd99",
    client_secret="9aa2f8b033104ec0b4092a685a6714b0",
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-read-playback-state,user-modify-playback-state,"
          "user-read-currently-playing,user-library-read,user-top-read,playlist-read-private",
    cache_path=".spotifycache"  # ✅ prevents repeated login prompts
))

# NLP & Face Pipelines
text_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
face_classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

# Mood map
mood_map = {
    "joy": "happy upbeat pop",
    "sadness": "emotional sad acoustic",
    "anger": "heavy rock metal",
    "fear": "calm piano instrumental",
    "neutral": "chill lofi hip hop",
    "surprise": "energetic edm",
    "disgust": "dark experimental",
}

# Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Speech Recognition
recognizer = sr.Recognizer()

class DemuAicAPI:
    """Python API for frontend."""

    def analyze_text(self, text):
        result = text_classifier(text)[0]
        mood = result["label"].lower()
        return self.get_track_for_mood(mood, confidence=result["score"])

    def analyze_face(self, image_data):
        """Detect face in base64 image and classify emotion."""
        try:
            # Decode base64 image
            img_bytes = base64.b64decode(image_data.split(",")[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                return {"error": "Could not decode image"}

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(60, 60))

            if len(faces) == 0:
                return {"error": "No face detected"}

            (x, y, w, h) = faces[0]  # first detected face
            face = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)

            preds = face_classifier(face_pil)
            mood = preds[0]["label"].lower()
            confidence = preds[0]["score"]

            return self.get_track_for_mood(mood, confidence=confidence)

        except Exception as e:
            return {"error": f"Face analysis failed: {str(e)}"}

    def analyze_voice(self):
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                text = recognizer.recognize_google(audio)
            return self.analyze_text(text)
        except Exception as e:
            return {"error": f"Voice recognition failed: {str(e)}"}

    def get_track_for_mood(self, mood, confidence=None):
        query = mood_map.get(mood, "chill music")
        results = sp.search(q=query, type="track", limit=5)["tracks"]["items"]
        if not results:
            return {"error": f"No track found for mood: {mood}"}

        chosen = random.choice(results)
        song_name = chosen["name"]
        artist = chosen["artists"][0]["name"]
        url = chosen["external_urls"]["spotify"]

        # Try to play on Spotify Premium; otherwise, open in browser
        try:
            devices = sp.devices()
            if devices["devices"]:
                device_id = devices["devices"][0]["id"]
                sp.start_playback(device_id=device_id, uris=[chosen["uri"]])
        except spotipy.exceptions.SpotifyException:
            # If playback fails (e.g., Free account), open in web browser
            webbrowser.open(url)

        return {
            "mood": mood,
            "confidence": round(confidence, 2) if confidence else None,
            "song": f"{song_name} - {artist}",
            "url": url
        }

if __name__ == "__main__":
    api = DemuAicAPI()
    window = webview.create_window("DemuAic - AI Mood Music", "home.html", js_api=api)
    webview.start(debug=True)

