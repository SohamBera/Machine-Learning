import pickle
import numpy as np
import cv2
import mediapipe as mp
# open the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)
# copy paste
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#set the gesture names for your labes as done in collected images
labels_dict = {0: 'Hello', 1: 'love', 2: 'Yes', 3: 'No', 4: 'Thank You', 5: 'Please', 6: 'Sorry', 7: 'eat', 8: 'drink', 9: 'Bye bye'}
while True:

    AUXILARY_IMAGE = []
    x_test = []
    y_test = []

    ret, frame_mirror = cap.read()
    frame = cv2.flip(frame_mirror, 1)

    H, W, _ = frame.shape

    color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(color_image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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

        x1 = int(min(x_test) * W) - 10
        y1 = int(min(y_test) * H) - 10

        x2 = int(max(x_test) * W) - 10
        y2 = int(max(y_test) * H) - 10

        prediction = model.predict([np.asarray(AUXILARY_IMAGE)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)

    key=cv2.waitKey(1)
    
    if(key==27):
        break

cap.release()
cv2.destroyAllWindows()