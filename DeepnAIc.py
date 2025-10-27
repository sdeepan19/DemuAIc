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
#           "user-read-currently-playing,user-library-read,user-top-read,playlist-read-private",
#     cache_path=".spotifycache"
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
#         """Detect face and return mood + Spotify track."""
#         try:
#             img_bytes = base64.b64decode(image_data.split(",")[1])
#             np_arr = np.frombuffer(img_bytes, np.uint8)
#             frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#             if frame is None:
#                 return {"error": "Could not decode image"}

#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(60, 60))
#             if len(faces) == 0:
#                 return {"error": "No face detected"}

#             (x, y, w, h) = faces[0]
#             face = frame[y:y+h, x:x+w]
#             face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#             face_pil = Image.fromarray(face_rgb)

#             preds = face_classifier(face_pil)
#             mood = preds[0]["label"].lower()
#             confidence = preds[0]["score"]

#             track_info = self.get_track_for_mood(mood)
#             return {
#                 "mood": mood,
#                 "confidence": round(confidence,2),
#                 "song": track_info["song"],
#                 "url": track_info["url"],
#                 "face_x": x,
#                 "face_y": y,
#                 "face_w": w,
#                 "face_h": h
#             }

#         except Exception as e:
#             return {"error": f"Face analysis failed: {str(e)}"}

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
#             return {"error": f"No track found for mood: {mood}"}

#         chosen = random.choice(results)
#         song_name = chosen["name"]
#         artist = chosen["artists"][0]["name"]
#         url = chosen["external_urls"]["spotify"]

#         try:
#             devices = sp.devices()
#             if devices["devices"]:
#                 device_id = devices["devices"][0]["id"]
#                 sp.start_playback(device_id=device_id, uris=[chosen["uri"]])
#         except spotipy.exceptions.SpotifyException:
#             webbrowser.open(url)

#         return {
#             "mood": mood,
#             "confidence": round(confidence,2) if confidence else None,
#             "song": f"{song_name} - {artist}",
#             "url": url
#         }

# if __name__ == "__main__":
#     api = DemuAicAPI()
#     window = webview.create_window("DemuAic - AI Mood Music", "home.html", js_api=api)
#     webview.start(debug=True)

    # client_id="8016d9d06cde4ed8a397f52d3c41dd99",
    # client_secret="9aa2f8b033104ec0b4092a685a6714b0",
import webview
import random
import webbrowser
from transformers import pipeline
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import cv2
import numpy as np
import base64
import time
from PIL import Image


# Spotify Authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="ENTER YOUR CLIENT ID",
    client_secret="ENTER YOUR CLIENT SECRET",
    redirect_uri="ENTER REDIRECT URL",
    scope="user-read-playback-state,user-modify-playback-state,"
          "user-read-currently-playing,user-read-playback-position,"
          "user-library-read,user-top-read,playlist-read-private"
))


# Hugging Face NLP Models
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
generator = pipeline("text-generation", model="gpt2")


# Mood-to-Spotify Query Mapping
mood_map = {
    "joy": "happy upbeat pop",
    "sadness": "emotional sad acoustic",
    "anger": "heavy rock metal",
    "fear": "calm piano instrumental",
    "neutral": "chill lofi hip hop",
    "surprise": "energetic edm",
    "disgust": "dark experimental",
}


def get_playlist_tracks(limit_playlists=20, limit_tracks_per_playlist=50):
    """
    Fetch tracks from the user's playlists.

    Args:
        limit_playlists (int): Number of playlists to fetch.
        limit_tracks_per_playlist (int): Number of tracks to fetch per playlist.

    Returns:
        list[str]: A list of track names from the user's playlists.
    """
    playlist_tracks = []
    try:
        playlists = sp.current_user_playlists(limit=limit_playlists)["items"]
        for pl in playlists:
            tracks = sp.playlist_items(pl["id"], limit=limit_tracks_per_playlist)["items"]
            for t in tracks:
                track = t.get("track")
                if track:
                    playlist_tracks.append(track["name"])
    except Exception as e:
        print(f"⚠️ Could not fetch playlist tracks: {e}")

    return playlist_tracks[:100]


def get_user_top_tracks_and_artists(limit=5):
    """
    Get the user's top artists and tracks.

    Args:
        limit (int): Number of top tracks to fetch.

    Returns:
        tuple[list[str], list[str]]: (Top artist names, Top track names).
    """
    try:
        top_tracks = sp.current_user_top_tracks(limit=limit, time_range="medium_term")["items"]
        top_artists = [t["artists"][0]["name"] for t in top_tracks]
        top_track_names = [t["name"] for t in top_tracks]
        return top_artists, top_track_names
    except:
        return [], []


def build_final_query(mood, user_artists, user_tracks, playlist_tracks):
    """
    Construct a Spotify search query based on:
    - Detected mood
    - User’s listening history (artists & tracks)
    - Playlist data
    - AI-generated extra query tokens

    Args:
        mood (str): Detected emotion label.
        user_artists (list[str]): User's top artists.
        user_tracks (list[str]): User's top tracks.
        playlist_tracks (list[str]): User's playlist track names.

    Returns:
        str: Final Spotify query string.
    """
    base_query = mood_map.get(mood, "chill music")

    prompt = f"Suggest one music style or artist for someone feeling {mood}."
    ai_output = generator(
        prompt, max_new_tokens=15, truncation=True,
        num_return_sequences=1, do_sample=True, top_k=50, top_p=0.9
    )[0]["generated_text"]

    ai_query = ai_output.replace(prompt, "").strip().split("\n")[0]
    ai_query = " ".join(ai_query.split()[:5])

    mode = random.choices(
        ["playlist", "top_artists", "top_tracks", "all_combined", "mood_only"],
        weights=[0.25, 0.3, 0.1, 0.4, 0.69],
        k=1
    )[0]

    if mode == "playlist":
        final_query = f"{base_query} {ai_query} {' '.join(playlist_tracks)}"[:240]
    elif mode == "top_artists":
        final_query = f"{base_query} {ai_query} {' '.join(user_artists)}"[:240]
    elif mode == "top_tracks":
        final_query = f"{base_query} {ai_query} {' '.join(user_tracks)}"[:240]
    elif mode == "all_combined":
        final_query = f"{base_query} {ai_query} {' '.join(playlist_tracks)} {' '.join(user_artists)} {' '.join(user_tracks)}"[:240]
    else:
        final_query = base_query

    return final_query


# Face Detection + Emotion Recognition
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
emotion_classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")


class API:
    """
    API bridge between frontend (index.html) and backend (Python).
    """

    def analyze_mood(self, text):
        """
        Analyze mood from user-provided text and play a recommended song.

        Args:
            text (str): Input sentence.

        Returns:
            dict: Contains mood, confidence, query, and song info.
        """
        result = classifier(text)[0]
        mood = result["label"].lower()

        playlist_tracks = get_playlist_tracks()
        user_artists, user_tracks = get_user_top_tracks_and_artists()

        final_query = build_final_query(mood, user_artists, user_tracks, playlist_tracks)
        tracks = sp.search(q=final_query, type="track", limit=5)["tracks"]["items"]

        if tracks:
            chosen = random.choice(tracks)
            song_name = chosen["name"]
            artist = chosen["artists"][0]["name"]
            url = chosen["external_urls"]["spotify"]
            uri = chosen["uri"]

            devices = sp.devices()
            if devices["devices"]:
                device_id = devices["devices"][0]["id"]
                sp.start_playback(device_id=device_id, uris=[uri])

            return {
                "mood": mood,
                "confidence": round(result["score"], 2),
                "query": final_query,
                "song": f"{song_name} - {artist}",
                "url": url
            }
        else:
            return {"error": "No matching track found."}

    def facial_recognition(self, image_data):
        """
        Detects a user's face from an image, classifies emotion,
        and plays a recommended Spotify track.

        Args:
            image_data (str): Base64-encoded image data (from webcam).

        Returns:
            dict: Contains mood, query, song info, and Spotify URL.
        """
        img_data = base64.b64decode(image_data.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Could not decode image"}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
        )

        if len(faces) == 0:
            return {"error": "No face detected."}

        (x, y, w, h) = faces[0]
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face_pil = Image.fromarray(face_rgb)
        preds = emotion_classifier(face_pil)
        mood = preds[0]["label"].lower()

        playlist_tracks = get_playlist_tracks()
        user_artists, user_tracks = get_user_top_tracks_and_artists()
        final_query = build_final_query(mood, user_artists, user_tracks, playlist_tracks)
        tracks = sp.search(q=final_query, type="track", limit=5)["tracks"]["items"]

        if not tracks:
            return {"error": "No track found."}

        chosen = random.choice(tracks)
        song_name = chosen["name"]
        artist = chosen["artists"][0]["name"]
        url = chosen["external_urls"]["spotify"]
        uri = chosen["uri"]

        devices = sp.devices()
        if devices["devices"]:
            device_id = devices["devices"][0]["id"]
            sp.start_playback(device_id=device_id, uris=[uri])

        return {
            "mood": mood,
            "query": final_query,
            "song": f"{song_name} - {artist}",
            "url": url
        }


if __name__ == "__main__":
    api = API()
    window = webview.create_window(
        "MusAIc - Feel the Music of Your Mood",
        url="index.html",
        js_api=api
    )

    def on_closed():
        """Exit the program cleanly when window is closed."""
        import sys
        sys.exit(0)

    window.events.closed += on_closed
    webview.start(debug=True)
