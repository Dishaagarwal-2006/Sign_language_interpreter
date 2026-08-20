import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# ---- SETTINGS ----
SHOW_IMAGES = True      # set to False to run silently (fast, no popup windows)
MAX_IMAGES_TO_SHOW = 5  # only show the first N images per class (avoids ~300 popups)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    print(f'Processing class {dir_}...')
    shown_count = 0

    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # draw landmarks on the image for visualization
                if SHOW_IMAGES and shown_count < MAX_IMAGES_TO_SHOW:
                    mp_drawing.draw_landmarks(
                        img_rgb,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            data.append(data_aux)
            labels.append(dir_)

            if SHOW_IMAGES and shown_count < MAX_IMAGES_TO_SHOW:
                plt.figure()
                plt.imshow(img_rgb)
                plt.title(f'Class {dir_} - {img_path}')
                plt.show()
                shown_count += 1

print(f'\nDone. Collected {len(data)} samples across {len(set(labels))} classes.')

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print('Saved to data.pickle')