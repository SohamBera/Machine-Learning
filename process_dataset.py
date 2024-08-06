import os
import cv2
import mediapipe as mp
import pickle
#generate mediapipe functions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

FILE_path = './imgs'

imgs = []
labels = []
for dir_ in os.listdir(FILE_path):
    for image_location in os.listdir(os.path.join(FILE_path, dir_)):
        AUXILARY_IMAGE = []

        x_test = []
        y_test = []

        image = cv2.imread(os.path.join(FILE_path, dir_, image_location))
        color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(color_image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_test.append(x)
                    y_test.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    AUXILARY_IMAGE.append(x - min(x_test))
                    AUXILARY_IMAGE.append(y - min(y_test))

            imgs.append(AUXILARY_IMAGE)
            labels.append(dir_)

file = open('imgs.pickle', 'wb')
pickle.dump({'imgs': imgs, 'labels': labels}, file)
file.close()



