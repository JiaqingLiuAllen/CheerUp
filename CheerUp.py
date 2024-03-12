import time
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Set up spotify api access
scope = "user-read-playback-state,user-modify-playback-state,playlist-read-private"
# Replace with your own client info on Spotify
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="YOUR_CLIENT_ID",
                                               client_secret="YOUR_CLIENT_SECRET",
                                               redirect_uri="http://localhost:8888/callback/",
                                               scope=scope))

# Different Spotify playlists to play when detecting different moods, using URIs
# For now, it doesn't play any music when detected emotion is neutral
emotion_playlists = {
    'Angry': 'spotify:playlist:37i9dQZF1EIgNZCaOGb0Mi',
    'Fear': 'spotify:playlist:37i9dQZF1EIfMwRYymgnLH',
    'Happy': 'spotify:playlist:37i9dQZF1EVJSvZp5AOML2',
    'Neutral': 'spotify:playlist:YourNeutralPlaylistURI',
    'Sad': 'spotify:playlist:37i9dQZF1EIdChYeHNDfK5',
    'Surprise': 'spotify:playlist:37i9dQZF1EIgTObt5JIqtf',
}


# Get dynamic device id (Spotify on a chrome page)
def get_active_device_id():
    devices = sp.devices()['devices']
    for device in devices:
        if device['is_active']:
            return device['id']  # Return the first active device
    return None  # No active device found


def play_playlist(playlist_uri):
    device_id = get_active_device_id()
    if device_id:
        try:
            sp.start_playback(device_id=device_id, context_uri=playlist_uri)
        except spotipy.exceptions.SpotifyException as e:
            print(f"Error playing playlist: {e}")
    else:
        print("No active device found")


# Load the pre-trained model
classifier = load_model('fer.h5')
# Define emotion labels according to the pre-trained model
class_labels = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
classes = list(class_labels.values())
face_classifier = cv2.CascadeClassifier('./Haarcascades/haarcascade_frontalface_default.xml')


# This function is for designing the overlay text on the predicted image boxes.
def text_on_detected_boxes(text,text_x,text_y,image,font_scale = 1,
                           font = cv2.FONT_HERSHEY_SIMPLEX,
                           FONT_COLOR = (0, 0, 0),
                           FONT_THICKNESS = 2,
                           rectangle_bgr = (0, 255, 0)):

    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]
    # Set the Coordinates of the boxes
    box_coords = ((text_x-10, text_y+4), (text_x + text_width+10, text_y - text_height-5))
    # Draw the detected boxes and labels
    cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(image, text, (text_x, text_y), font, fontScale=font_scale, color=FONT_COLOR,thickness=FONT_THICKNESS)


# Detection of the emotions on an image:
def face_detector_image(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) # Convert the image into GrayScale image
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img

    allfaces = []
    rects = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        allfaces.append(roi_gray)
        rects.append((x, w, y, h))
        

def emotionImage(imgPath):
    img = cv2.imread(imgPath)
    rects, faces, image = face_detector_image(img)

    i = 0
    for face in faces:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # make a prediction on the ROI, then lookup the class
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]
        label_position = (rects[i][0] + int((rects[i][1] / 2)), abs(rects[i][2] - 10))
        i = + 1

        # Overlay our detected emotion on the picture
        text_on_detected_boxes(label, label_position[0],label_position[1], image)

    cv2.imshow("Emotion Detector", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Detection of the expression on video stream
def face_detector_video(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        roi_gray = gray[y:y + h, x:x + w]

    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    return (x, w, y, h), roi_gray, img


def emotionVideo(cap):
    last_emotion = "Neutral"  # Initialize last_emotion as "Neutral" to allow immediate playback for any other emotion
    last_change_time = time.time() - 300  # Subtract 300 seconds to allow immediate change if a non-neutral emotion is detected

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        rect, face, image = face_detector_video(frame)
        if np.sum([face]) != 0.0:
            roi = face.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = classifier.predict(roi)[0]
            current_emotion = class_labels[preds.argmax()]
            label_position = (rect[0] + rect[1]//50, rect[2] + rect[3]//50)

            text_on_detected_boxes(current_emotion, label_position[0], label_position[1], image)

            current_time = time.time()
            # Change playlist only if the detected emotion is not "Neutral", is different from the last non-neutral emotion,
            # and at least 5 minutes have passed since the last playlist change.
            if (current_emotion != "Neutral" and current_emotion != last_emotion and
                (current_time - last_change_time >= 300)):
                try:
                    play_playlist(emotion_playlists[current_emotion])
                    last_emotion = current_emotion
                    last_change_time = current_time
                except spotipy.exceptions.SpotifyException as e:
                    print(f"Error playing playlist: {e}")
            elif current_emotion == "Neutral":
                # For future designs to handle "Neutral" differently
                pass
        else:
            cv2.putText(image, "No Face Found", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Emotion Detector', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def list_devices():
    devices = sp.devices()
    print("Available devices:")
    for device in devices['devices']:
        print(f"ID: {device['id']}, Name: {device['name']}, Type: {device['type']}, Active: {device['is_active']}")

if __name__ == '__main__':
    list_devices()  # List available devices to find the correct device_id
    camera = cv2.VideoCapture(0)
    # camera = cv2.VideoCapture('PATH_TO_THE_VIDEO') # If you are feeding a video in the current directory as this file.
    emotionVideo(camera)

