import cv2
import os
FILE_path = './imgs'
# create a file path if its already not exists
if not os.path.exists(FILE_path):
    os.makedirs(FILE_path)

gestures = 10 #{0: 'Hello', 1: 'love', 2: 'Yes', 3: 'No', 4: 'Thank You', 5: 'Please', 6: 'Sorry', 7: 'eat', 8: 'drink', 9: 'Bye bye'}
no_of_collected_images = 50

cap = cv2.VideoCapture(0)
for n in range(gestures):
    if not os.path.exists(os.path.join(FILE_path, str(n))):
        os.makedirs(os.path.join(FILE_path, str(n)))
    print('Collecting Images for gesture : {}'.format(n))
    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Press `S` to start...', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 102, 204), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('s'):
            break
    count = 0
    while count < no_of_collected_images:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(FILE_path, str(n), '{}.jpg'.format(count)), frame)

        count += 1

cap.release()
cv2.destroyAllWindows()